from setuptools import setup, find_packages, Extension
setup( name='LMPC_walking',
       version='1.0.1',
       description='robust generation for humanoid robots using LIPM',
       packages=['second_order', 'second_order/mpc', 'second_order/rmpc',
       'second_order/stmpc'],  zip_safe=True)
