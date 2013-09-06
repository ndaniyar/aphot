import sys
from numpy import *
from common import sdict

def tryeval(_s):
    try: _s = eval(_s)
    except (SyntaxError, NameError, AttributeError): pass
    return _s

def tryfloat(s):
    try: s = float(s)
    except ValueError: pass
    return s


def get_args(**defaults):
    ''' --arg_name=value OR pos_arg1 pos_arg2
    '''
    # first look at the command line
    cmd_args = {}
    pos = []
    for arg in sys.argv:
        if arg[0:2] == '--':
            key, val = arg.lstrip('-').split('=', 1)
            cmd_args[key] = tryeval(val)
        else:
            pos += [tryeval(arg)]

    # and then prioritize the entries
    cfg_file = cmd_args.get('cfg', defaults.get('cfg', None))
    args = {}
    if cfg_file is not None: # Use cfg file
        args['cfg'] = cfg_file
        execfile(cfg_file, args)
        del args['__builtins__']
        # args in the cfg file is merged to the function argumet args
        # positional is overriden if it exists in cfg file

    # Override file's entries with command line arguments
    args.update(cmd_args)

    positional = ['script_name'] + args.get('positional',[])
    if len(positional) == len(pos):
        args.update(zip(positional,pos))
    else:
        raise Exception('Wrong number of positional arguments')

    defaults.update(args)
    return sdict(defaults)
