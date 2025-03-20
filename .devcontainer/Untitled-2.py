ERROR: test_order_manager (unittest.loader._FailedTest.test_order_manager)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_order_manager
Traceback (most recent call last):
  File "C:\Program Files\Python313\Lib\unittest\loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "C:\Program Files\Python313\Lib\unittest\loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "D:\BROKR\tests\test_order_manager.py", line 4, in <module>
    from trading_execution.order_manager import OrderManager
ModuleNotFoundError: No module named 'trading_execution.order_manager'


----------------------------------------------------------------------
Ran 2 tests in 0.001s

FAILED (errors=2)