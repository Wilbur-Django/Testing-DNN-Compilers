import time
import os
import random


class TimeIterator:
    def __init__(self, iterator, time_recording_file):
        self.gen_order = list(range(len(iterator)))
        random.shuffle(self.gen_order)
        self.gen_list = list(iterator)
        self.time_list = [-1000 for _ in range(len(self.gen_order))]
        self.time_rec_file = time_recording_file
        self.cur_iter = -1
        self.st_time = None

    def __len__(self):
        return len(self.gen_order)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_iter >= len(self.gen_order) - 1:
            with open(self.time_rec_file, 'w') as f:
                sorted_list = sort_timing(self.gen_list, self.time_list)
                f.write("\n".join([f"{name}$$${t}" for name, t in sorted_list if t != -1000]))
            # plot_time(self.time_rec_file)
            raise StopIteration
        else:
            self.cur_iter += 1
            print("==================")
            print("Compiling (running):",
                  self.gen_list[self.gen_order[self.cur_iter]])
            return self.gen_list[self.gen_order[self.cur_iter]]

    def set_time(self, run_time):
        self.time_list[self.gen_order[self.cur_iter]] = run_time

    def cal_time(self, lambda_func, repeat_times=1):
        st_time = time.time()
        r = lambda_func()
        for i in range(repeat_times - 1):
            lambda_func()
        elapsed_time = time.time() - st_time
        self.set_time(elapsed_time / repeat_times)
        return r


def time_iterator(iterator, time_file):
    return TimeIterator(iterator, time_file)


def plot_time(time_file):
    from matplotlib import pyplot as plt
    # Cannot plot on X shell
    with open(time_file, 'r') as f:
        name_time_list = [line.strip().split("$$$") for line in f.readlines()]
    name_time_list = [(int(n), float(t)) if n != 'seed' else (0, float(t))
                      for n, t in name_time_list]
    plt.plot(*zip(*name_time_list))

    save_name = "%s.png" % (os.path.splitext(os.path.basename(time_file))[1])
    save_path = os.path.join(os.path.dirname(time_file), save_name)
    plt.savefig(save_path)


def sort_timing(name_list, time_list):
    d = list(range(len(name_list)))
    d.sort(key=lambda x: int(name_list[x]) if name_list[x] != 'seed' else 0)
    return [(name_list[t], time_list[t]) for t in d]
