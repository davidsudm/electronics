#!/usr/bin/env python
"""
Read scope files and do their waveforms

Usage:
    music-readout --input_dir=PATH --output_dir=PATH --channel=N --polarization=N [--debug]

Options:
    -h -help                    Show this screen.
    --input_dir=PATH            Path to the input directory, where the input files are located.
    --output_dir=PATH           Path to the output directory, where the outputs (pdf files) will be saved.
    --channel=N                 Channel number of MUSIC
    --polarization=N            Polarization of the waveform, either +1 or -1
    -v --debug                  Enter the debug mode.
"""

import os
import re
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from docopt import docopt
from digicampipe.utils.docopt import convert_int, convert_text


def give_list_of_file(input_dir):

    files = [f for f in os.listdir(input_dir) if
             os.path.isfile(os.path.join(input_dir, f)) and not f.startswith('.')]

    files.sort()

    return files


def read_data(file, start_event=0, polarization=1):
    has_found_data = False
    event_count = 0
    with open(file, 'r') as f:
        for line in f:
            if'Npoints:' in line:
                n_points = int(line.split(': ')[1])
            if 'Increment:' in line:
                increment = float(line.split(': ')[1])
            if 'Waveform:' in line:
                has_found_data = True
                event_count += 1
                continue
            if has_found_data and (event_count >= start_event):
                time = np.arange(0, n_points * increment, increment)
                waveform = np.array(line.split(' ')[:-1], dtype=float)
                waveform *= polarization
                has_found_data = False

                yield time, waveform


def sum_waveforms(file, polarization=1):

    waveforms = read_data(file=file, polarization=polarization)
    wave_sum = 0.0
    cnt = 0
    for time, waveform in waveforms:
        time = time[:5000]
        waveform = waveform[:5000]
        wave_sum += waveform
        cnt += 1

    print('Total number of waveforms : ', cnt)
    return time, wave_sum, cnt


def average_baseline(file, polarization=1):

    _time, summed_waveform, events = sum_waveforms(file, polarization)

    peak_index = np.argmax(summed_waveform)
    baseline = summed_waveform[:peak_index - 200] / events
    baseline = np.mean(baseline)

    return baseline, peak_index


def draw_waveform(a_time, a_waveform, peak_index=None, baseline=None, label=None):

    if baseline is not None:
        a_waveform = (a_waveform - baseline)
        a_time = a_time
    else:
        a_waveform = a_waveform

    if peak_index is not None:
        a_waveform = a_waveform[peak_index - 150: peak_index + 1200]
        a_time = a_time[peak_index - 150: peak_index + 1200]
    else:
        a_waveform = a_waveform

    a_time = a_time / 1e-6
    a_waveform = a_waveform * 1e3

    fig, ax = plt.subplots()
    if label is not None:
        ax.plot(a_time, a_waveform, label=label)
    else:
        ax.plot(a_time, a_waveform)

    ax.set_xlabel(r'Time [$\mu s$]')
    ax.set_ylabel('Amplitude [mV]')

    if label is not None:
        ax.legend()

    return fig, ax


def entry():
    args = docopt(__doc__)
    input_dir = convert_text(args['--input_dir'])
    output_dir = convert_text(args['--output_dir'])
    channel = convert_int(args['--channel'])
    polarization = convert_int(args['--polarization'])
    debug = args['--debug']

    file_list = give_list_of_file(input_dir)

    for f in file_list:

        bias_voltage = float(re.findall('\d+\.\d+', f)[0])
        f = input_dir + '/' + f
        print(f)

        pdf_waveforms_draw = PdfPages('{}/waveforms_ch{}_{}V.pdf'.format(output_dir, channel, bias_voltage))

        baseline, peak_index = average_baseline(file=f, polarization=polarization)
        time, sum, cnt = sum_waveforms(file=f, polarization=polarization)
        fig, ax = draw_waveform(a_time=time, a_waveform=sum, label='Summed waveforms : {} events'.format(cnt))
        pdf_waveforms_draw.savefig(fig)
        plt.close(fig)

        if debug:
            print('Drawing last {} waveforms'.format(100))
            start_event = cnt - 100

            waveforms = read_data(file=f, polarization=polarization, start_event=start_event)

            for time, waveform in waveforms:
                fig, ax = draw_waveform(a_time=time, a_waveform=waveform, baseline=baseline, peak_index=peak_index)
                pdf_waveforms_draw.savefig(fig)
                plt.close(fig)
                fig, ax = draw_waveform(a_time=time, a_waveform=waveform, baseline=baseline)
                pdf_waveforms_draw.savefig(fig)
                plt.close(fig)

        pdf_waveforms_draw.close()
        print('PDF saved in : {}/waveforms_ch{}_{}V.pdf'.format(output_dir, channel, bias_voltage))


if __name__ == '__main__':
    entry()

