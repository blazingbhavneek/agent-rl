from helpers.time_it import time_it

@time_it()
def time_function():
    '''doc-string for time_function.'''
    import time
    time.sleep(2)
    print('slept for 2 seconds.')
    return

time_function()
print(time_function.__doc__)