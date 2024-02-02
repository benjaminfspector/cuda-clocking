import argparse
import numpy as np
import re
import tabulate
np.set_printoptions(suppress=True, precision=4)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='profile.csv', help='Path to the profile file')
parser.add_argument('-s', '--src', default=None, help='Path to the source file')

args = parser.parse_args()

profile_file = args.input
with open(profile_file, 'r') as file:
    filename = file.readline()
    srcfile = filename.strip() if args.src is None else args.src.strip()
    data = np.array([list(map(int, x.split(',')[:-1])) for x in file.read().strip().split('\n')])

# open source file
with open(srcfile, 'r') as file:
    names, lines = {}, {}
    for linenumber, line in enumerate(file.readlines()):
        matches = re.findall(r'CLOCKPOINT\(.*\)', line)
        if len(matches) == 0 or '(ID, STRING...)' in matches[0]:
            continue
        contents = matches[0][11:-1].split(',')
        id = int(contents[0])
        name = ','.join(contents[1:]).strip()
        if ',' not in name:
            name = name.replace('"', '').strip()
        names[id] = name
        lines[id] = linenumber + 1
    for i in range(100):
        if i not in names:
            names[i] = f'UNK<{i}>'
            lines[i] = f'UNK<{i}>'

# clock data
count_data = data[:, ::2]
clock_data = data[:, 1::2]

total_data = clock_data.astype(np.float32)
per_iter_data = total_data / count_data

total_per_point = np.mean(total_data, axis=0)
cum_percent = total_per_point / np.sum(total_per_point) * 100
counts_per_point = np.mean(count_data.astype(np.float32), axis=0)
mean_per_iter = np.mean(per_iter_data, axis=0)
std_per_iter = np.std(per_iter_data, axis=0)

sortedtotals = sorted([(total_per_point[i], i) for i in range(len(total_per_point))], reverse=True)

print(tabulate.tabulate(
    [(x[1], f'{names[x[1]]}', f'{lines[x[1]]}', f"{cum_percent[x[1]]:.2f}%", f"{x[0]:.2f}", f"{counts_per_point[x[1]]:.2f}", f"{mean_per_iter[x[1]]:.2f}", f"{std_per_iter[x[1]]:.2f}") for x in sortedtotals if counts_per_point[x[1]]>0],
    headers=['ID #', 'Name', 'Line #', 'Frac', 'Mean cycles', 'Mean Count', 'Mean cycles/iter', 'Stddev cycles/iter'],
    stralign="right",
    disable_numparse=True
))