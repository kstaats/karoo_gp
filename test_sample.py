import os, time, sys
# from subprocess import PIPE
import pexpect

def log_stdout(ch, fname='log.txt'):
    '''Copy stdout lines to log file'''
    msg = ch.before.decode('utf-8').splitlines()
    with open(fname, 'a') as f:
        for line in msg:
            f.write(line + '\n')

def test_init():
    '''Runs a basic simulation. If no errors are thrown, it went as expected.'''
    logfile = 'log.txt'
    with open(logfile, 'w') as f:
        f.write('')
    ch = pexpect.spawn('python3 karoo-gp.py')
    
    # Select model type
    ch.expect('default m', timeout=8)
    log_stdout(ch, fname=logfile)
    ch.sendline('c')

    # Select tree style
    ch.expect('default r', timeout=1),
    log_stdout(ch, fname=logfile)
    ch.sendline('\n')

    # Enter initial depth
    ch.expect('default 3', timeout=1)
    log_stdout(ch, fname=logfile)
    ch.sendline('\n')

    # Enter max depth
    ch.expect('default 3', timeout=1)
    log_stdout(ch, fname=logfile)
    ch.sendline('\n')

    # Enter min nodes
    ch.expect('max 15', timeout=1)
    log_stdout(ch, fname=logfile)
    ch.sendline('\n')

    # Enter trees per population
    ch.expect('default 100', timeout=1)
    log_stdout(ch, fname=logfile)
    ch.sendline('10')
    
    # Enter max generations
    ch.expect('default 10', timeout=1)
    log_stdout(ch, fname=logfile)
    ch.sendline('3')

    # Display mode
    ch.expect('default m', timeout=1)
    log_stdout(ch, fname=logfile)
    ch.sendline('\n')

    ch.expect('pause', timeout=10)
    log_stdout(ch, fname=logfile)
    print('done')