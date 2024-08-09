is_simple = True
if is_simple:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable
    from dezero.core_simple import no_grad
else:
    None

setup_variable()