import sys
from pywebcachesim import parser, get_task, runner


def main():
    # f string is supported in 3.6
    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
        raise Exception('Error: python version need to be at least 3.6')
    args = parser.parse_cmd_args()
    tasks = get_task.get_task(args)
    if args["debug"]:
        print(tasks)
    return runner.run(args, tasks)


if __name__ == '__main__':
    main()
