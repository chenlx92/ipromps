#!/usr/bin/python
import load_data
import train_models


def main():
    print('## Running the %s' % load_data.__name__)
    load_data.main()
    print('## Running the %s' % train_models.__name__)
    train_models.main()

if __name__ == '__main__':
    main()
