
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    
    import mmaction.datasets
    import mmaction.engine
    import mmaction.evaluation
    import mmaction.models
    import mmaction.structures
    import mmaction.visualization

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None\
                        or not DefaultScope.check_instance_created('mmaction')
        if never_created:
            DefaultScope.get_instance('mmaction', scope_name='mmaction')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmaction':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmaction", '
                          '`register_all_modules` will force set the current'
                          'default scope to "mmaction". If this is not as '
                          'expected, please set `init_default_scope=False`.')

            new_instance_name = f'mmaction-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmaction')
