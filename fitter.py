#!/usr/bin/env python
"""
Read scope files and do their waveforms

Usage:
    music-fit multigauss --input_dir=PATH --output_dir=PATH --channel=N [--debug]
    music-fit digicampipe --input_dir=PATH --initial_values_dir=PATH --output_dir=PATH --channel=N [--debug]

Options:
    -h -help                    Show this screen.
    --input_dir=PATH            Path to the input directory, where the input files are located.
    --output_dir=PATH           Path to the output directory, where the outputs (pdf files) will be saved.
    --initial_values_dir=PATH   Path to the directory containing YML files (dictionaries) containing the initialization values for the fit
    --channel=N                 Channel number used in MUSIC.
    -v --debug                  Enter the debug mode.

Commands:
    multigauss                  Multi-Gauss peak fitter method
    digicampipe                 digicampipe fitter method for multi-photon spectra

"""

import os
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.pyplot import *
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
from scipy import optimize
from scipy.optimize import curve_fit
from scipy import interpolate

from docopt import docopt
from digicampipe.utils.docopt import convert_int, convert_text
from histogram.histogram import Histogram1D
from digicampipe.scripts.mpe import fit_single_mpe
import readout as read


def gauss(x, A, mu, sigma):
    """
    Gaussian function

    :param x:           x values
    :param A:           Amplitude (actually, number of entries, area under the curve)
    :param sigma:       standard deviation of the gaussianinit_A1
    :param mu:          mean value of the gaussian
    :return:            Gaussian pdf
    """
    norm = sigma * np.sqrt(2 * np.pi)

    return A * (1 / norm) * np.exp(-(((x - mu) / sigma) ** 2) / 2)


def multi_gauss(x, A1=0, A2=0, A3=0, x_peak=0, gain=0, sigma_e=0, sigma_s=0):

    A = [A1, A2, A3]
    sigma = [sigma_e, sigma_e + sigma_s, sigma_e + 2*sigma_s]
    mu = [x_peak, x_peak + gain, x_peak + 2*gain]

    m_gauss = gauss(x, A[0], mu[0], sigma[0]) + gauss(x, A[1], mu[1], sigma[1]) + gauss(x, A[2], mu[2], sigma[2])

    return m_gauss


# def multi_gauss_sum(x, n_peaks, A1=0, A2=0, A3=0, x_peak=0, gain=0, sigma_e=0, sigma_s=0):
#
#     for k, peak in enumerate(n_peaks):
#         if k < 100:
#             gauss(x, A[k], mu[k], sigma[k])
#
#     A = [A1, A2, A3]
#     sigma = [sigma_e, sigma_e + sigma_s, sigma_e + 2*sigma_s]
#     mu = [x_peak, x_peak + gain, x_peak + 2*gain]
#
#     m_gauss = gauss(x, A[0], mu[0], sigma[0]) + gauss(x, A[1], mu[1], sigma[1]) + gauss(x, A[2], mu[2], sigma[2])
#
#     return m_gauss


def write_gaus_info(var=[0, 0, 0], var_err=[0, 0, 0]):

    decimal = 2

    for k, variable in enumerate(var):
        var[k] = np.around(var[k], decimals=decimal)
        var_err[k] = np.around(var_err[k], decimals=decimal)

    text = ' A : {} ± {}\n'\
           ' $\mu$ : {} ± {}\n'\
           ' $\sigma$ : {} ± {}'.format(var[0], var_err[0],
                                        var[1], var_err[1],
                                        var[2], var_err[2]
                                        )
    return text


def write_multi_gaus_info(var=[0, 0, 0, 0, 0, 0, 0], var_err=[0, 0, 0, 0, 0, 0, 0]):

    decimal = 2

    for k, variable in enumerate(var):
        var[k] = np.around(var[k], decimals=decimal)
        var_err[k] = np.around(var_err[k], decimals=decimal)

    text = ' $A_0$ : {} ± {}\n'\
           ' $A_1$ : {} ± {}\n'\
           ' $A_2$ : {} ± {}\n'\
           ' $\mu_1$ : {} ± {}\n'\
           ' Gain : {} ± {}\n'\
           ' $\sigma_e$ : {} ± {}\n'\
           ' $\sigma_s$ : {} ± {}'.format(var[0], var_err[0],
                                          var[1], var_err[1],
                                          var[2], var_err[2],
                                          var[3], var_err[3],
                                          var[4], var_err[4],
                                          var[5], var_err[5],
                                          var[6], var_err[6],
                                          )

    return text


def fit_gaussian_peak(x, y, bin_size=1, display_offset=0):

    init_amplitude = np.sum(y) * bin_size
    init_mu = np.sum(y * x) / np.sum(y)
    init_sigma = np.sqrt(np.sum(y * (x - init_mu) ** 2) / np.sum(y))

    popt, pcov = curve_fit(gauss, x, y, p0=[init_amplitude, init_mu, init_sigma])
    var = popt
    var_err = np.sqrt(np.diagonal(pcov))
    text = write_gaus_info(var, var_err)

    x_fit = np.linspace(x[0], x[-1], 1000)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(x + display_offset, y, 'b-', label='data')
    ax0.plot(x_fit + display_offset, gauss(x_fit, *popt), 'g--', label='fit')
    ax0.plot(x + display_offset, gauss(x, init_amplitude, init_mu, init_sigma), 'r*', label='Initial values')
    ax0.set_ylabel('count')
    ax0.legend(loc=2)

    text_gaus = 'y = A $\\frac{1}{\sigma \sqrt{2 \pi}} e^{-(\\frac{x-\mu}{\sigma})}$'
    anchored_text = AnchoredText(text, loc=6, frameon=False)
    anchored_text_gaus = AnchoredText(text_gaus, loc=1, frameon=False)
    ax0.add_artist(anchored_text)
    ax0.add_artist(anchored_text_gaus)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(x + display_offset, (y - gauss(x, *popt)) / y, marker='o', linestyle='None', color='black')
    ax1.axhline(0, color='gray', linestyle='dashed')
    ax1.set_ylabel('Residual')
    ax1.set_xlabel('Index')
    print('Peak fitted')

    return popt, pcov, fig


def entry():

    args = docopt(__doc__)
    input_dir = convert_text(args['--input_dir'])
    output_dir = convert_text(args['--output_dir'])
    channel = convert_int(args['--channel'])
    initial_values_dir = convert_text(args['--initial_values_dir'])
    debug = args['--debug']

    if args['digicampipe']:

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
        print('END of the digicampipe fitter')

    if args['multigauss']:

        file_list = read.give_list_of_file(input_dir)
        file_list.sort()
        print(file_list)

        # Creating the dictionary
        fit_parameters = {}

        for k, f in enumerate(file_list):

            f = os.path.join(input_dir, f)
            bias_voltage = float(re.findall('\d+\.\d+', f)[0])

            charge_histogram = Histogram1D.load(f)
            y_data = charge_histogram.data
            x_data = charge_histogram.bin_centers

            if debug:
                pdf_debug_histo_to_plot = PdfPages(os.path.join(output_dir, 'ch{}_debug_histo_to_plot_V{}.pdf'.format(bias_voltage, channel)))
                fig, (ax1, ax2) = plt.subplots(2, 1)
                charge_histogram.draw(axis=ax1, legend=False, label='histogram data')
                ax1.plot(x_data, y_data, '|', label='plotting of data', mec='tab:orange', mfc='tab:orange',
                         markersize=12)
                ax1.set_xlabel('[LSB]')
                ax1.legend()

                ax2.plot(y_data, label='plotting of data', marker='|', color='tab:blue', mfc='tab:orange',
                         mec='tab:orange', markersize=12)
                ax2.set_ylabel('count')
                ax2.set_xlabel('Index')
                ax2.legend()
                pdf_debug_histo_to_plot.savefig(fig)
                pdf_debug_histo_to_plot.close()
                plt.close(fig)

            # Automatizing the initial values guess
            # Find peaks: Gain
            # First approx, it will set us to the separation of to peaks
            idx_peaks, _ = find_peaks(y_data, height=20)
            delta_idx_peak = np.diff(idx_peaks)
            idx_peak_bis, _ = find_peaks(y_data, distance=delta_idx_peak[0], height=40)
            print('idx of peaks found', idx_peak_bis)
            print('Peaks found : ', len(idx_peak_bis))

            if debug:
                pdf_debug_peaks_found = PdfPages(os.path.join(output_dir, 'ch_{}_debug_peaks_found_V{}.pdf'.format(bias_voltage, channel)))
                fig, (ax1, ax2) = plt.subplots(2, 1)

                ax1.plot(y_data, color='tab:blue', label='Integral charge')
                ax1.plot(idx_peaks, y_data[idx_peaks], color='tab:green',
                         label='Peaks found : 1st round', marker='v', linestyle='None')
                ax1.set_xlabel('Index')
                ax1.set_ylabel('count')
                ax1.legend()

                ax2.plot(y_data, color='tab:blue', label='Integral charge')
                ax2.plot(idx_peak_bis, y_data[idx_peak_bis], color='tab:red',
                         label='Peaks found : 2st round', marker='v', linestyle='None')
                ax2.set_xlabel('Index')
                ax2.set_ylabel('count')
                ax2.legend()
                pdf_debug_peaks_found.savefig(fig)
                pdf_debug_peaks_found.close()
                plt.close(fig)

            initial_values = []
            # We can do better : fitting the first peak with a gaussian to extract initial parameters
            # Defining the first peak (peak zeros)
            # safe_zone expansion
            safe_zone = 3
            idx_valley = np.argmin(y_data[idx_peak_bis[0]: idx_peak_bis[1]])
            idx_valley += idx_peak_bis[0]

            interval = [0, idx_valley]
            y_peak = y_data[interval[0]: interval[1] + safe_zone]
            x_peak = np.arange(len(y_peak))

            pdf_fit_peak = PdfPages(os.path.join(output_dir, 'ch{}_fit_peak_V{}.pdf'.format(bias_voltage, channel)))

            popt, pcov, fig = fit_gaussian_peak(x_peak, y_peak)
            pdf_fit_peak.savefig(fig)
            plt.show()

            initial_values.append(popt)

            # Defining the second peak
            # idx_valley is key since new fit will be shift by the quantity idx_valley (important for plot)
            interval = [idx_valley, idx_valley + delta_idx_peak[0]]
            y_peak = y_data[interval[0]: interval[-1] + safe_zone]
            x_peak = np.arange(len(y_peak))

            popt, pcov, fig = fit_gaussian_peak(x_peak, y_peak, display_offset=interval[0])
            popt[1] += interval[0]
            pdf_fit_peak.savefig(fig)
            plt.show()

            initial_values.append(popt)

            # Defining the third peak
            # idx_valley is key since new fit will be shift by the quantity idx_valley (important for plot)
            interval = [interval[0] + delta_idx_peak[0], interval[0] + 2*delta_idx_peak[0]]
            y_peak = y_data[interval[0]: interval[-1] + safe_zone]
            x_peak = np.arange(len(y_peak))

            popt, pcov, fig = fit_gaussian_peak(x_peak, y_peak, display_offset=interval[0])
            popt[1] += interval[0]
            pdf_fit_peak.savefig(fig)
            plt.show()

            initial_values.append(popt)

            pdf_fit_peak.close()

            print('Initial values gotten')
            initial_values = np.array(initial_values)

            amplitude = initial_values.T[0]
            x_first_peak = initial_values.T[1][0]
            gain = np.diff(initial_values.T[1])[0]
            sigma_e = initial_values.T[2][0]
            sigma_s = np.sqrt(initial_values.T[2][1]**2 - sigma_e**2)

            # Fitting the multi-gauss function of 3 peaks
            x_data = np.arange(len(y_data))
            popt, pcov = curve_fit(multi_gauss, x_data, y_data, p0=[amplitude[0], amplitude[1], amplitude[2],
                                                                    x_first_peak, gain, sigma_e, sigma_s])

            var = popt
            var_err = np.sqrt(np.diagonal(pcov))
            text = write_multi_gaus_info(var, var_err)

            x_fit = np.linspace(x_data[0], x_data[-1], 1000)

            fig = plt.figure()
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            ax0 = fig.add_subplot(gs[0])
            ax0.plot(x_data, y_data, 'b-', label='data')
            ax0.plot(x_fit, multi_gauss(x_fit, *popt), 'g--', label='fit')
            ax0.plot(x_data, multi_gauss(x_data, amplitude[0], amplitude[1], amplitude[2], x_first_peak, gain, sigma_e, sigma_s), 'r*', label='Initial values', ms=2)
            ax0.set_ylabel('count')
            ax0.legend(loc=9)

            text_formula = 'y =  $ \\sum_{k=0}^{N=2} A_k\\frac{1}{\sigma_k \sqrt{2 \pi}} e^{-(\\frac{x-\mu_k}{\sigma_k})}$\n' \
                           ' $\sigma_k^2 = \sigma_e^2 + k\sigma_s^2$'
            anchored_text = AnchoredText(text, loc=1, frameon=False)
            anchored_formula = AnchoredText(text_formula, loc=4, frameon=False)
            ax0.add_artist(anchored_text)
            ax0.add_artist(anchored_formula)

            ax1 = fig.add_subplot(gs[1], sharex=ax0)
            ax1.plot(x_data, (y_data - multi_gauss(x_data, *popt)) / y_data, marker='o', ms=4, linestyle='None', color='black')
            ax1.axhline(0, color='gray', linestyle='dashed')
            ax1.set_ylabel('Residual')
            ax1.set_xlabel('Index')
            print('Multi-Gauss fitted')

            pdf_fit_multigauss = PdfPages(os.path.join(output_dir, 'ch{}_fit_multigauss_V{}.pdf'.format(bias_voltage, channel)))
            pdf_fit_multigauss.savefig(fig)
            pdf_fit_multigauss.close()
            plt.show()
            plt.close(fig)

            # Creating the sub-dictionary
            level = 'LVL_{}'.format(k)

            temp_dict = {}
            temp_dict['bias_voltage'] = bias_voltage
            temp_dict['amplitude'] = np.sum(var[0:2])
            temp_dict['mu_peak'] = var[3]
            temp_dict['gain'] = var[4]
            temp_dict['sigma_e'] = var[5]
            temp_dict['sigma_s'] = var[6]

            temp_dict['error_amplitude'] = np.sqrt(np.sum(var_err[0:2]**2))
            temp_dict['error_mu_peak'] = var_err[3]
            temp_dict['error_gain'] = var_err[4]
            temp_dict['error_sigma_e'] = var_err[5]
            temp_dict['error_sigma_s'] = var_err[6]

            fit_parameters[level] = temp_dict
            del temp_dict

            print('Multi-Gauss fitter for voltage {} V done '.format(bias_voltage))

        if debug:
            print('fit_parameter', fit_parameters)
        fit_parameters_file = os.path.join(output_dir, 'fit_parameters.yml')

        with open(fit_parameters_file, 'w') as file:
            yaml.dump(fit_parameters, file)

        print('Fitted parameters saved at : {}'.format(fit_parameters_file))
        print('END of the Multi-Gauss fitter')


if __name__ == '__main__':
    entry()
