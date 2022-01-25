
import os

# for some akward reason, put scripts before unittests tests  
scripts = ['debug.py',
           'hydraulic.py',
           # Unit tests after that
           ]

for script_name in scripts:
    print('\n## Executing script {}'.format(script_name))
    exec(open(script_name).read())
