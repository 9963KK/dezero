is_simple = False
if is_simple:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable
    from dezero.core_simple import no_grad
else:
    from dezero.core import Variable
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable
    from dezero.core import no_grad
    import dezero.functions
    import dezero.utils
    from dezero.layers import Layer
    from dezero.models import Model
setup_variable()