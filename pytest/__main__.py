import sys
import subprocess


def main():
    # Filter out pytest-style flags like -q or -s
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if not args:
        args = ['discover', '-s', 'tests']
    cmd = [sys.executable, '-m', 'unittest'] + args
    sys.exit(subprocess.call(cmd))

if __name__ == '__main__':
    main()
