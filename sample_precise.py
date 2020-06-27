#!/usr/bin/env python3

from precise_runner import PreciseRunner, PreciseEngine
from threading import Event
from time import sleep


def main():
    # parser = ArgumentParser('Implementation demo of precise-engine')
    # parser.add_argument('engine', help='Location of binary engine file')
    # parser.add_argument('model')
    # args = parser.parse_args()

    def on_prediction(prob):
        print('!' if prob > 0.5 else '.')

    def on_activation():
        print("activation!")
        # activate_notify()

    # engine = PreciseEngine(args.engine, args.model)
    # engine = PreciseEngine('/home/rbccps2080ti/.virtualenvs/precise/bin/precise-engine', '/home/rbccps2080ti/projects/speech/wake-word-benchmark/audio/computer/hey-computer.net')
    engine = PreciseEngine('/home/rbccps2080ti/.virtualenvs/precise/bin/precise-engine', '/home/rbccps2080ti/projects/speech/wake-word-benchmark/audio/computer/hey-computer.net')
    p = PreciseRunner(engine, on_prediction=on_prediction, on_activation=on_activation,
                  trigger_level=0)
    p.start()
    sleep(10)
    p.stop()

if __name__ == '__main__':
    main()
