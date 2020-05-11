#!/usr/bin/env python
"""
Read scope files and do their waveforms

Usage:
    music-charge compute --input_dir=PATH --output_dir=PATH --channel=N --polarization=INT [--debug]
    music-charge fit --input_dir=PATH --initial_values_dir=PATH --output_dir=PATH --channel=N [--debug]
    music-charge save_figure --input_dir=PATH --output_dir=PATH --channel=N [--debug]

Options:
    -h -help                    Show this screen.
    --input_dir=PATH            Path to the input directory, where the input files are located.
    --output_dir=PATH           Path to the output directory, where the outputs (pdf files) will be saved.
    --initial_values_dir=PATH   Path to the directory containing YML files (dictionaries) containing the initialization values for the fit
    --fit_parameters_dir=PATH   Path to the directory containing the fitted parameters dictionary obtained 'from music-charge fit'.
    --channel=N                 Channel number used in MUSIC.
    --polarization=N            Polarization of the waveform. If waveform is negative, multiple by -1, else, +1
    -v --debug                  Enter the debug mode.

Commands:
    compute                     compute charge histograms and save them as fits files
    fit                         fit histograms from fits files using digicampipe methods
    save_figure                 makes histograms and fits figures
"""

import re
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks

from docopt import docopt
from digicampipe.utils.docopt import convert_int, convert_text
from histogram.histogram import Histogram1D
from digicampipe.scripts.mpe import fit_single_mpe
import readout as read


def get_amplitude_charge(waveform_iter, peak_index, baseline=None, window_bound=[0, 0]):
    """

    :param waveform_iter:       waveform iterator
    :param peak_index:          index of the waveform peak
    :param baseline:            baseline of the waveform computer before the peak
    :param window_bound:        left and right bins away from the peak to evaluate the maximum amplitude
    :return:                    Maximum of the waveform inside the window's bound, index of the left and right window's
                                boundary
    """

    print('Amplitude charge')

    left_window_bound = peak_index - window_bound[0]
    right_window_bound = peak_index + window_bound[1]

    amplitude_charge = []

    if baseline is None:
        baseline_excess = 0.0
    else:
        baseline_excess = baseline

    for time, waveform in waveform_iter:

        waveform = waveform - baseline_excess

        time /= 1e-6
        waveform *= 1e3

        max_amplitude = np.max(waveform[left_window_bound: right_window_bound])
        amplitude_charge.append(max_amplitude)

    amplitude_charge = np.array(amplitude_charge)

    return amplitude_charge, left_window_bound, right_window_bound


def get_integral_charge(waveform_iter, peak_index, baseline=None, integration_bounds=[0, 0]):
    """

    :param waveform_iter:       waveform iterator
    :param peak_index:          index of the waveform peak
    :param baseline:            baseline of the waveform computer before the peak
    :param integration_bounds:  left and right bins away from the peak to do the sum
    :return:                    Integral charge, index of the left integration boundary, index of the right integration
                                boundary
    """

    print('Integral charge')

    left_integration_bound = peak_index - integration_bounds[0]
    right_integration_bound = peak_index + integration_bounds[1]

    integral_charge = []

    if baseline is None:
        baseline_excess = 0.0
    else:
        baseline_excess = baseline

    for time, waveform in waveform_iter:

        waveform -= baseline_excess

        # Factor multiplication is to go from V to mV
        # Factor multiplication is to go from s to µs
        time /= 1e-6
        waveform *= 1e3

        charge = time * waveform
        charge = np.sum(charge[left_integration_bound: right_integration_bound])
        integral_charge.append(charge)

    integral_charge = np.array(integral_charge)

    return integral_charge, left_integration_bound, right_integration_bound


def get_peaks(charge, prominence=None, threshold=None, width=None, distance=None, debug=None):

    peaks, _ = find_peaks(charge, prominence=prominence, threshold=threshold, width=width, distance=distance)
    plt.plot(peaks, charge[peaks], "ob")
    if debug:
        print('Peaks', peaks)
        print('Charge[peaks]', charge[peaks])
    plt.plot(charge)
    plt.legend(['prominence'])
    plt.show()

    x_peak = peaks
    y_peaks = charge[peaks]

    return x_peak, y_peaks


def entry():

    args = docopt(__doc__)
    input_dir = convert_text(args['--input_dir'])
    output_dir = convert_text(args['--output_dir'])
    channel = convert_int(args['--channel'])
    polarization = convert_int(args['--polarization'])
    initial_values_dir = convert_text(args['--initial_values_dir'])
    debug = args['--debug']

    if args['compute']:

        file_list = read.give_list_of_file(input_dir)

        pdf_charge_draw = PdfPages('{}/plots/charge_ch{}.pdf'.format(output_dir, channel))

        for f in file_list:
            bias_voltage = float(re.findall('\d+\.\d+', f)[0])
            f = input_dir + '/' + f
            print(f)

            baseline, peak_index = read.average_baseline(file=f, polarization=polarization)

            waveforms = read.read_data(file=f, polarization=polarization)
            integral_charge, left_int_bound, right_int_bound = get_integral_charge(waveform_iter=waveforms,
                                                                                   peak_index=peak_index,
                                                                                   baseline=baseline,
                                                                                   integration_bounds=[10, 15])
            waveforms = read.read_data(file=f, polarization=polarization)
            amplitude_charge, left_window_bound, right_window_bound = get_amplitude_charge(waveform_iter=waveforms,
                                                                                           peak_index=peak_index,
                                                                                           baseline=baseline,
                                                                                           window_bound=[50, 100])

            if debug:

                print('Charge debugging waveform')
                waveforms = read.read_data(file=f, polarization=polarization)
                cnt = 0

                pdf_charge_debug = PdfPages('{}/plots/debug_charge_ch{}_V{}.pdf'.format(output_dir, channel, bias_voltage))

                for time, waveform in waveforms:

                    if cnt % 1000 == 0:
                        fig, ax = read.draw_waveform(time, waveform, baseline=baseline, label='waveform {}'.format(cnt))
                        time = time / 1e-6
                        waveform = waveform * 1e3
                        ax.plot(time[left_int_bound: right_int_bound],
                                waveform[left_int_bound: right_int_bound],
                                label='Integration samples', color='tab:red')
                        ax.legend()

                        pdf_charge_debug.savefig(fig)
                        plt.close(fig)
                    cnt += 1

                pdf_charge_debug.close()
                print('End of charge debugging waveform')

            histo_data = [integral_charge, amplitude_charge]
            histo_label = ['integral charge', 'amplitude charge']

            for i, data in enumerate(histo_data):

                # Histogram creation
                bins = 100
                bin_edges = np.linspace(np.min(data), np.max(data) + 1, bins)
                histogram = Histogram1D(bin_edges=bin_edges)
                histogram.fill(data)

                # Histogram display
                fig, ax = plt.subplots()
                histogram.draw(axis=ax, label='{} : V = {} V'.format(histo_label[i], bias_voltage), legend=False)
                text = histogram._write_info(())
                anchored_text = AnchoredText(text, loc=5)
                ax.add_artist(anchored_text)
                pdf_charge_draw.savefig(fig)
                print('{} at {} V figure saved'.format(histo_label[i], bias_voltage))
                #plt.show()
                plt.close(fig)

                # Formatting to use in the digicampipe fitting single mpe method
                histogram.data = histogram.data.reshape((1, 1, -1))
                histogram.overflow = histogram.overflow.reshape((1, -1))
                histogram.underflow = histogram.underflow.reshape((1, -1))

                histogram.save('{}/charge/{}/ch{}_V_{}.fits'.format(output_dir,
                                                                    histo_label[i].replace(" ", "_"),
                                                                    channel,
                                                                    bias_voltage))

        pdf_charge_draw.close()

    if args['fit']:

        file_list = read.give_list_of_file(input_dir)
        yaml_list = read.give_list_of_file(initial_values_dir)

        file_list.sort()
        yaml_list.sort()

        print(file_list)
        print(yaml_list)

        fit_parameters = {}

        for k, f in enumerate(file_list):

            level = 'LVL_{}'.format(k)
            bias_voltage = float(re.findall('\d+\.\d+', f)[0])

            print('Fitting charge')
            f = input_dir + '/' + f
            i_val = initial_values_dir + '/' + yaml_list[k]
            print('charge file :', f)
            print('initialization file :', i_val)

            with open(i_val) as file:
                init_parameters = yaml.load(file, Loader=yaml.FullLoader)
                print('Initial Fitting parameters', init_parameters)

            # We need this new format to make work our fit function, it was built that way
            temp_dict = {}
            for key, value in init_parameters.items():
                temp_dict[key] = np.array([[value]])
            init_parameters = temp_dict
            del temp_dict

            data = fit_single_mpe(f, ac_levels=[0], pixel_ids=[0], init_params=init_parameters, debug=True)

            temp_data = {}
            for key, value in data.items():
                if key is not 'pixel_ids':
                    temp_data[key] = (value[0][0]).tolist()

            temp_data['bias_voltage'] = bias_voltage
            fit_parameters[level] = temp_data

        print('fit_parameter', fit_parameters)
        fit_parameters_file = '{}/fit_parameters.yml'.format(output_dir)

        with open(fit_parameters_file, 'w') as file:
            yaml.dump(fit_parameters, file)

    if args['save_figure']:
        print('save_figure')

        file_list = read.give_list_of_file(input_dir)

        fig, ax = plt.subplots()
        for f in file_list:
            bias_voltage = float(re.findall('\d+\.\d+', f)[0])
            f = os.path.join(input_dir, f)
            print('File : ', f)

            histogram = Histogram1D.load(f)

            histogram.draw(axis=ax, legend=False, label='Bias voltage : {}'.format(bias_voltage))

        pdf_superposition_charge = PdfPages(os.path.join(output_dir, 'charge_in_bias_voltage_ch{}.pdf'.format(channel)))
        pdf_superposition_charge.savefig(fig)
        pdf_superposition_charge.close()
        #plt.show()
        plt.close(fig)


if __name__ == '__main__':
    entry()
