#!/usr/bin/env python
"""
Compute the breakdown voltage from bias voltage and mpe fit scope files and do their waveforms

Usage:
    music-breakdown compute --input_dictionary=FILE --output_dir=PATH --channel=N --method=STR [--debug]

Options:
    -h -help                    Show this screen.
    --input_dictionary=FILE     Path to the input directory, where the input files are located.
    --output_dir=PATH           Path to the output directory, where the outputs (pdf files) will be saved.
    --channel=N                 Channel number of MUSIC.
    --method=STR                String, either amplitude or integral.
    -v --debug                  Enter the debug mode.

Commands:
    compute                     compute breakdown, make plots and save them as fits files
"""

import re
import yaml
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages

from docopt import docopt
from digicampipe.utils.docopt import convert_int, convert_text


def plot_linear_fit_(x, y, err_y, channel, chi_2=None, chi_tolerance=0):

    x = np.array(x)
    y = np.array(y)
    err_y = np.array(err_y)
    chi_2 = np.array(chi_2)

    if chi_2 is not None:
        if chi_tolerance is not 0:
            mask = chi_2 <= chi_tolerance
            x = x[mask]
            y = y[mask]
            err_y = err_y[mask]

    (m, b), cov = np.polyfit(x, y, 1, cov=True, w=1/err_y, full=False)
    error_m = np.sqrt(cov[0][0])
    error_b = np.sqrt(cov[1][1])

    x_fit = np.linspace(np.min(x), np.max(x), 1000)
    y_fit = m * x_fit + b

    breakdown_voltage = - b/m
    #error_breakdown_voltage = np.sqrt((m**-2 * b * error_m)**2 + (error_b / m)**2 - 2*(error_b/b)*(error_m/m)*1)
    #error_breakdown_voltage = breakdown_voltage * np.sqrt((error_b/b)**2 + (error_m/m)**2)
    error_breakdown_voltage = np.sqrt((error_b/m)**2 + (b*error_m/m**2)**2 + 2*(error_b/m)*(b*error_m/m**2))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    print('slope : {:.2f} ± {:.2f} V'.format(m, error_m))
    print('intersect : {:.2f} ± {:.2f} V'.format(b, error_b))
    print('breakdown voltage : {:.2f} ± {:.2f} V'.format(breakdown_voltage, error_breakdown_voltage))

    text = ' y = m $V_{{bias}}$ + b \n' \
           ' $V_{{break}}$ : {:.2f} ± {:.2f} [V] \n' \
           ' m : {:.2f} ± {:.2f} [LSB / V] \n' \
           ' b : {:.2f} ± {:.2f} [LSB]'.format(breakdown_voltage, error_breakdown_voltage, m, error_m, b, error_b)

    fig, ax = plt.subplots()

    ax.errorbar(x, y, yerr=err_y, label='CH{} : data'.format(channel), linestyle=' ', marker='o', ms=3, color='tab:green', lolims=True, uplims=True, capthick=1)
    ax.plot(x_fit, y_fit, label='fit', color='tab:red', linestyle='dashed')
    ax.set_xlabel(r'$V_{bias}$ [V]')
    ax.legend(loc=4, fancybox=True, framealpha=0.5)
    anchored_text = AnchoredText(text, loc=2, pad=0.8, frameon=False)
    ax.add_artist(anchored_text)

    return fig, ax


def entry():

    args = docopt(__doc__)
    input_dictionary = convert_text(args['--input_dictionary'])
    output_dir = convert_text(args['--output_dir'])
    channel = convert_int(args['--channel'])
    method = convert_text(args['--method'])
    debug = args['--debug']

    with open(input_dictionary) as file:
        parameters_per_bias_voltage = yaml.load(file, Loader=yaml.FullLoader)

        if debug:
            print('Initial Fitting parameters', parameters_per_bias_voltage)

    if args['compute']:

        level = []

        amplitude = []
        baseline = []
        bias_voltage = []
        chi_2 = []
        error_amplitude = []
        error_baseline = []
        error_gain = []
        error_mu = []
        error_mu_xt = []
        error_n_peaks = []
        error_sigma_e = []
        error_sigma_s = []
        gain = []
        mean = []
        mu = []
        mu_xt = []
        n_peaks = []
        ndf = []
        sigma_e = []
        sigma_s = []
        std = []

        for key, value in parameters_per_bias_voltage.items():

            sub_dictionary = parameters_per_bias_voltage[key]

            level.append(int(re.findall('\d+', key)[0]))
            amplitude.append(sub_dictionary['amplitude'])
            baseline.append(sub_dictionary['baseline'])
            bias_voltage.append(sub_dictionary['bias_voltage'])
            chi_2.append(sub_dictionary['chi_2'])
            error_amplitude.append(sub_dictionary['error_amplitude'])
            error_baseline.append(sub_dictionary['error_baseline'])
            error_gain.append(sub_dictionary['error_gain'])
            error_mu.append(sub_dictionary['error_mu'])
            error_mu_xt.append(sub_dictionary['error_mu_xt'])
            error_n_peaks.append(sub_dictionary['error_n_peaks'])
            error_sigma_e.append(sub_dictionary['error_sigma_e'])
            error_sigma_s.append(sub_dictionary['error_sigma_s'])
            gain.append(sub_dictionary['gain'])
            mean.append(sub_dictionary['mean'])
            mu.append(sub_dictionary['mu'])
            mu_xt.append(sub_dictionary['mu_xt'])
            n_peaks.append(sub_dictionary['n_peaks'])
            ndf.append(sub_dictionary['ndf'])
            sigma_e.append(sub_dictionary['sigma_e'])
            sigma_s.append(sub_dictionary['sigma_s'])
            std.append(sub_dictionary['std'])

        pdf_breakdown_draw = PdfPages(os.path.join(output_dir, 'ch{}_breakdown.pdf'.format(channel)))

        reduce_chi2 = np.array(chi_2) / np.array(ndf)

        if method == 'amplitude':
            tolerance = 1e9
        elif method == 'integral':
            tolerance = 1e9
        else:
            tolerance = 0

        fig, ax = plot_linear_fit_(bias_voltage, gain, error_gain, channel=channel, chi_2=reduce_chi2, chi_tolerance=tolerance)
        ax.set_ylabel('Gain [LSB]')
        pdf_breakdown_draw.savefig(fig)
        plt.close(fig)

        pdf_breakdown_draw.close()

        if debug:

            print('level', level)
            print('amplitude', amplitude)
            print('baseline', baseline)
            print('bias_voltage', bias_voltage)
            print('chi_2', chi_2)
            print('error_amplitude', error_amplitude)
            print('error_baseline', error_baseline)
            print('error_gain', error_gain)
            print('error_mu', error_mu)
            print('error_mu_xt', error_mu_xt)
            print('error_n_peaks', error_n_peaks)
            print('error_sigma_e', error_sigma_e)
            print('error_sigma_s', error_sigma_s)
            print('gain', gain)
            print('mean', mean)
            print('mu', mu)
            print('mu_xt', mu_xt)
            print('n_peaks', n_peaks)
            print('ndf', ndf)
            print('sigma_e', sigma_e)
            print('sigma_s', sigma_s)
            print('std', std)

            pdf_breakdown_debug = PdfPages('{}/ch{}_breakdown_debug.pdf'.format(output_dir, channel))

            fig, ax = plt.subplots()
            ax.plot(bias_voltage, level, label='Level', marker='o', linestyle='None', color='tab:green')
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel('Light Level')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.errorbar(bias_voltage, amplitude, error_amplitude, label='Amplitude', marker='o', linestyle='None', color='tab:green', lolims=True, uplims=True)
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel('Amplitude [counts]')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.errorbar(bias_voltage, baseline, error_baseline, label='Baseline', marker='o', linestyle='None', color='tab:green', lolims=True, uplims=True)
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel('Baseline [LSB]')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(bias_voltage, bias_voltage, label='Bias voltage', marker='o', linestyle='None', color='tab:green')
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel(r'$V_{bias}$ [V]')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(bias_voltage, chi_2, label=r'$\chi^2$', marker='o', linestyle='None', color='tab:green')
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel(r'$\chi^2$')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.errorbar(bias_voltage, gain, error_gain, label='Gain', marker='o', linestyle='None', color='tab:green', lolims=True, uplims=True)
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel('Gain [LSB / p.e.]')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.errorbar(bias_voltage, mean, std, label='Mean', marker='o', linestyle='None', color='tab:green', lolims=True, uplims=True)
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel('Mean [counts]')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.errorbar(bias_voltage, mu, error_mu, label=r'$\mu$', marker='o', linestyle='None', color='tab:green', lolims=True, uplims=True)
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel(r'$\mu$ [p.e.]')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.errorbar(bias_voltage, mu_xt, error_mu_xt, label=r'$\mu_{XT}$', marker='o', linestyle='None', color='tab:green', lolims=True, uplims=True)
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel(r'$\mu_{XT}$ [p.e.]')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.errorbar(bias_voltage, n_peaks, error_n_peaks, label='Number of peaks', marker='o', linestyle='None', color='tab:green', lolims=True, uplims=True)
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel('Number of peaks []')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(bias_voltage, ndf, label='ndf', marker='o', linestyle='None', color='tab:green')
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel('ndf []')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.errorbar(bias_voltage, sigma_e, error_sigma_e, label=r'$\sigma_e$', marker='o', linestyle='None', color='tab:green', lolims=True, uplims=True)
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel('$\sigma_e$ [LSB]')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.errorbar(bias_voltage, sigma_s, error_sigma_s, label=r'$\sigma_s$', marker='o', linestyle='None', color='tab:green', lolims=True, uplims=True)
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel('$\sigma_s$ [LSB]')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(bias_voltage, np.array(chi_2)/np.array(ndf), label=r'$\chi^2$ / ndf', marker='o', linestyle='None', color='tab:green')
            ax.set_xticks(bias_voltage)
            ax.set_xlabel(r'$V_{bias}$ [V]')
            ax.set_ylabel(r'$\chi^2$ / ndf')
            ax.legend(loc=4, fancybox=True, framealpha=0.5)
            pdf_breakdown_debug.savefig(fig)
            plt.close(fig)

            pdf_breakdown_debug.close()


if __name__ == '__main__':
    entry()