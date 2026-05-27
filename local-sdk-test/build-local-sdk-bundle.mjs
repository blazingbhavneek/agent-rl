import { builtinModules, createRequire } from "node:module";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..");
const sdkPackagesRoot = resolve(repoRoot, "cline/sdk/packages");
const buildRoot = resolve(__dirname, ".local-sdk-builder");
const builderPackageJsonPath = join(buildRoot, "package.json");
const localSdkPackageRoot = join(buildRoot, "package");
const localSdkPackageJsonPath = join(localSdkPackageRoot, "package.json");

const packageNames = ["shared", "llms", "agents", "core", "sdk"];

function isLocalClinePackage(name) {
  return name.startsWith("@cline/");
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

async function collectExternalDependencies() {
  const dependencies = {};

  for (const packageName of packageNames) {
    const packageJson = await readJson(
      join(sdkPackagesRoot, packageName, "package.json"),
    );

    for (const sectionName of ["dependencies", "peerDependencies"]) {
      const section = packageJson[sectionName] || {};
      for (const [name, version] of Object.entries(section)) {
        if (isLocalClinePackage(name)) {
          continue;
        }
        dependencies[name] ||= version;
      }
    }
  }

  dependencies.esbuild ||= "^0.25.5";
  return dependencies;
}

async function prepareBuilderFiles() {
  const externalDependencies = await collectExternalDependencies();

  await mkdir(buildRoot, { recursive: true });
  await mkdir(localSdkPackageRoot, { recursive: true });

  await writeFile(
    builderPackageJsonPath,
    `${JSON.stringify(
      {
        name: "local-sdk-bundle-builder",
        private: true,
        type: "module",
        dependencies: externalDependencies,
      },
      null,
      2,
    )}\n`,
    "utf8",
  );

  await writeFile(
    localSdkPackageJsonPath,
    `${JSON.stringify(
      {
        name: "@cline/sdk",
        version: "0.0.42-local",
        private: true,
        main: "./dist/index.cjs",
        exports: {
          ".": "./dist/index.cjs",
        },
      },
      null,
      2,
    )}\n`,
    "utf8",
  );
}

function getAliasMap() {
  return {
    "@cline/core": resolve(sdkPackagesRoot, "core/src/index.ts"),
    "@cline/agents": resolve(sdkPackagesRoot, "agents/src/index.ts"),
    "@cline/llms": resolve(sdkPackagesRoot, "llms/src/index.ts"),
    "@cline/shared": resolve(sdkPackagesRoot, "shared/src/index.ts"),
    "@cline/shared/storage": resolve(
      sdkPackagesRoot,
      "shared/src/storage/index.ts",
    ),
    "@cline/shared/db": resolve(sdkPackagesRoot, "shared/src/db/index.ts"),
    "@cline/shared/automation": resolve(
      sdkPackagesRoot,
      "shared/src/automation/index.ts",
    ),
    "@cline/shared/remote-config": resolve(
      sdkPackagesRoot,
      "shared/src/remote-config/index.ts",
    ),
  };
}

function getNodeExternals() {
  return Array.from(
    new Set([
      ...builtinModules,
      ...builtinModules.map((moduleName) => `node:${moduleName}`),
    ]),
  );
}

async function buildBundle() {
  if (!existsSync(builderPackageJsonPath)) {
    throw new Error(
      `Missing ${builderPackageJsonPath}. Run "node build-local-sdk-bundle.mjs prepare" first.`,
    );
  }

  const requireFromBuilder = createRequire(builderPackageJsonPath);
  const esbuild = requireFromBuilder("esbuild");

  await mkdir(join(localSdkPackageRoot, "dist"), { recursive: true });

  await esbuild.build({
    entryPoints: [resolve(sdkPackagesRoot, "agents/src/index.ts")],
    outfile: join(localSdkPackageRoot, "dist/index.cjs"),
    absWorkingDir: buildRoot,
    bundle: true,
    format: "cjs",
    platform: "node",
    target: "node22",
    alias: getAliasMap(),
    external: getNodeExternals(),
    nodePaths: [join(buildRoot, "node_modules")],
    logLevel: "info",
    legalComments: "none",
    sourcemap: false,
  });
}

async function main() {
  const mode = process.argv[2] || "prepare";

  if (mode === "prepare") {
    await prepareBuilderFiles();
    process.stdout.write(`${buildRoot}\n`);
    return;
  }

  if (mode === "bundle") {
    await buildBundle();
    process.stdout.write(`${localSdkPackageRoot}\n`);
    return;
  }

  throw new Error(`Unknown mode: ${mode}`);
}

await main();
