class Runnable:
    def __init__(self, func):
        self.func = func

    def __or__(self, other):
        def chained_func(*args, **kwargs):
            # the other func consumes the result of this func
            return other(self.func(*args, **kwargs))

        return Runnable(chained_func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def add_five(x):
    return x + 5


def multiply_by_two(x):
    return x * 2


# wrap the functions with Runnable
add_five = Runnable(add_five)
multiply_by_two = Runnable(multiply_by_two)

# run them using the object approach
chain = add_five | multiply_by_two
print(chain(3))
