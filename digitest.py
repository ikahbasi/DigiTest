import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from obspy import UTCDateTime as utc
import glob
import os
import scipy
from scipy import interpolate
from obspy import read
from scipy import fftpack


def find_near_value_index(array, value):
    a = np.abs(np.array(array)-value)
    idx = a.argmin()
    return idx
###############################################################################
def seiscomp(folder, format='miniseed'):
    '''
    this function is for make seiscomp directory of file that this code can
    work with it.
    if you have data that get form internal digitizer, you have to run this
    function before any work.
    like:
    '''
    stream = read('{}/*.{}'.format(folder, format))
    juldays = []
    for tr in stream:
        sday = tr.stats.starttime.julday
        eday = tr.stats.endtime.julday
        for day in range(sday, eday+1):
            juldays.append(day)
    juldays = list(set(juldays))
    year = tr.stats.starttime.year
    for julday in juldays:
        stime = utc(year=year, julday=julday, hour=0, minute=0, second=0)
        etime = utc(year=year, julday=julday, hour=23, minute=59, second=59,
                    microsecond=990000)
        st = stream.slice(stime, etime)
        stations = []
        channels = []
        networks = []
        for tr in st:
            stations.append(tr.stats.station)
            channels.append(tr.stats.channel)
            networks.append(tr.stats.network)
        stations = list(set(stations))
        channels = list(set(channels))
        networks = list(set(networks))
        for network in networks:
            for station in stations:
                for channel in channels:
                    print(network, station, channel)
                    st2save = stream.select(station=station,
                                            channel=channel,
                                            network=network)
                    if len(st2save)!=0:
                        path = os.path.join('..',
                                            str(year),
                                            network,
                                            station,
                                            channel+'.D')
                        filename = f'{network}.{station}..{channel}.D.{year}.{day}'
                        os.makedirs(path, exist_ok=True)
                        outfile = f'{path}/{filename}'
                        if os.path.isfile(outfile):
                            st2save.write(outfile+'new', format='MSEED')
                            print(outfile+'\tNEWFILE')
                        else:
                            st2save.write(outfile, format='MSEED')
                            print(outfile)


###############################################################################
def auto_read_data(inp, stream=None):
    # if it's first time that want to read data
    # if new data is not same as pervious data
    stime = utc(inp.starttime)
    sday = stime.julday
    year = utc(inp.starttime).year
    etime = stime + inp.duration or utc(inp.endtime)
    eday = etime.julday
    if (stream is None) or\
        (stream[0].stats.network != inp.network) or\
        (stream[0].stats.station != inp.station) or\
        (stream[0].stats.channel != inp.channel) or\
        (stream[0].stats.endtime.julday != eday) or\
        (stream[0].stats.starttime.julday != sday):
        print('Read from seiscomp directory')
        print(inp)
        from obspy import Stream
        stream = Stream()
        for day in range(sday, eday+1):
            # path of file( . is archive)
            path = os.path.join(
                    '..', str(year), inp.network, inp.station, f'{inp.channel}.D')
            name = f'{inp.network}.{inp.station}..{inp.channel}.D.{year}.{day}'
            # find file
            path = glob.glob(f'{path}/{name}')[0]
            # if file not exist then return 0
            if not os.path.isfile(path):
                print(path, ' NOT EXICT')
                return 0
            st = read(path)
            for tr in st:
                stream += tr
    return stream


###############################################################################
def _finalise_figure(fig, **kwargs):  # pragma: no cover
    """
    Internal function to wrap up a figure.
    {plotting_kwargs}
    """
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    show = kwargs.get("show", True)
    save = kwargs.get("save", False)
    savefile = kwargs.get("savefile", "EQcorrscan_figure.png")
    title = kwargs.get("title")
    xlim = kwargs.get("xlim")
    ylim = kwargs.get("ylim")
    if title:
        plt.title(title, fontsize=25)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(top=ylim[1])
    if save:
        path = os.path.dirname(savefile)
        if path:
            os.makedirs(path, exist_ok=True)
    return_fig = kwargs.get("return_figure", False)
    size = kwargs.get("size", (10.5, 7.5))
    fig.set_size_inches(size)
    if save:
        fig.savefig(savefile, bbox_inches="tight", dpi=130)
        print("Saved figure to {0}".format(savefile))
    if show:
        plt.show(block=True)
    if return_fig:
        return fig
    fig.clf()
    plt.close(fig)
    return None


###############################################################################
def _get_time_error(tr, mean=None, show=False):
    stime = tr.stats.starttime
    etime = tr.stats.endtime
    if stime.microsecond != 0:
        tr = tr.slice(stime+(1-stime.microsecond*0.000001), etime)
        print('start time become correcte now')
    else:
        print('start time was ok')
    # make array of data and time
    data = tr.data
    time_tr = np.arange(0,
                        tr.stats.npts * tr.stats.delta,
                        tr.stats.delta)
    if time_tr.size > data.size:
        time_tr = np.delete(time_tr, -1)
    # midele of y axis
    if mean is None:
        mean = (data.max() + data.min())/2
    # function of time trace and data
    func = interpolate.UnivariateSpline(time_tr, data, s=0)
    # reduce array of data. point of mean become zeros(roots) of array
    data_reduced = data - mean
    # make function for new array to finde root
    func_reduced = interpolate.UnivariateSpline(time_tr, data_reduced, s=0)
    # find value of x that makes root of function
    x_y_roots_data_reduced = func_reduced.roots()
    # for find true point
    # which point we are need it?
    deriv = func.derivatives(x_y_roots_data_reduced[0])[1]
    if deriv > 0:
        n = 0
    elif deriv < 0:
        n = 1
    x_y_roots_data_reduced = x_y_roots_data_reduced[n:-1:2]
    y = func(x_y_roots_data_reduced)
    # plot data and point
    if show:
        date = stime.strftime('%Y-%m-%d')
        stime = stime.strftime('%H:%M:%S')
        etime = etime.strftime('%H:%M:%S')
        plt.plot(time_tr, data, 'g', x_y_roots_data_reduced, y, '.r', linewidth=0.5)
        title = f'ID: {tr.id}\n{date} ({stime} to {etime})'
        plt.title(title, fontweight='black')
        plt.show()
    # find resisual of time error
    time_errors = x_y_roots_data_reduced - x_y_roots_data_reduced.round()
    return time_errors


###############################################################################
def _RMS(array):
    rms = np.sqrt(np.mean(np.square(array.astype('int64'))))
    return rms


###############################################################################
def PreProcessStream(stream):
    stream.print_gaps()
    stream.detrend('constant')
    stream.merge(fill_value=0)


###############################################################################
class DynamicRange:
    def __init__(self):
        self.id = None
        self.sampling_rate = None
        self.mean = None
        self.rms_n_detrend = None
        self.rms_y_detrend = None
        self.dynamicrange_n_detrend = None
        self.dynamicrange_y_detrend = None


###############################################################################
class freq_amp_psd:
    def __init__(self, _id):
        self.id = _id
        self.frequency = []
        self.amplitude = []
    
    def plot(self, method='semilogx'):
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.semilogx(self.frequency, self.amplitude, label=self.id)
        plt.ylabel('PSD [V**2/Hz dB]', fontweight='black')
        plt.title('Power Spectral Density', fontweight='black')
        plt.xlabel('Frequency [Hz]', fontweight='black')
        plt.legend()
        plt.grid()
        plt.show()


###############################################################################
class freq_amp:
    def __init__(self, _id, TYPE):
        self.id = _id
        self.frequency = []
        self.amplitude = []
        self.TYPE = TYPE
    def plot(self, **kwargs):
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.semilogx(self.frequency, self.amplitude, label=self.id)
        plt.ylabel('Amplitude [V dB]', fontweight='black')
        plt.title(f'Frequency Response\n({self.TYPE} input)', fontweight='black')
        plt.xlabel('Frequency [Hz]', fontweight='black')
        plt.legend()
        plt.grid()
        _finalise_figure(fig, **kwargs)


###############################################################################
class TimeError_part:
    def __init__(self, counts, bins, time_errors, starttime, endtime):
        self.bins = bins
        self.counts = counts
        self.time_errors = time_errors
        self.starttime = starttime
        self.endtime = endtime


###############################################################################
class TimeError:
    def __init__(self):
        self.parts = {}
        self.all_errors = None
        self.trend = []
        self.len_part = None
        self.starttime = None
        self.endtime = None
        

###############################################################################
class result:
    def __init__(self, _id):
        self.id = _id
        self.details = []
        self.ppsd = freq_amp(_id, TYPE='')
        self.freqresp_singelfreq = freq_amp(_id, TYPE='Singel frequency')
        self.freqresp_whitenoise = freq_amp(_id, TYPE='White noise')
        self.gain = None
        self.psd = freq_amp_psd(_id)
        self.dynamic_range = DynamicRange()
        self.time_error = TimeError()
   
###############################################################################
class CheckDigitizer:
    def __init__(self):
        self.inputs = None
        self.results = {}
        self.dynamic_range_table = None
        self.gain_table = None


    ###########################################################################
    def input_parser(self, input_file):
        inp = pd.read_csv(input_file, delim_whitespace=True, na_values='-',
                          comment='#')
        self.inputs = inp.fillna(method='ffill')
    
    ###########################################################################
    def update_results(self, **kwargs):
        if self.key not in self.results.keys():
            self.results[self.key] = result()
    
    ###########################################################################
    def psd(self, stream=None):
        for indx, inp in self.inputs.iterrows():
            if stream is None:
                stream = auto_read_data(inp, stream)
            st = stream.select(network=inp.network,
                               station=inp.station,
                               channel=inp.channel)
            if 'endtime' not in inp.keys():
                print(inp.duration)
                inp.endtime = utc(inp.starttime) + (inp.duration*60)
            st = st.slice(starttime=utc(inp.starttime),
                          endtime=utc(inp.endtime))
            PreProcessStream(st)
            tr = st[0]
            data = tr.data / inp.gain
            sps = tr.stats.sampling_rate
            freq, psd = signal.welch(x=data, fs=sps, nperseg=sps*3600)
            psd = 20 * np.log10(psd)                                           # mutiple 10 convert to 20 (1400-03-09)
            key = f'{tr.id}-{int(inp.sampling_rate)}sps-{inp.vpp}vpp'
            if key not in self.results.keys():
                self.results[key] = result(key)
            self.results[key].psd.frequency = freq
            self.results[key].psd.amplitude = psd
            
    ###########################################################################
    def plot_psd(self, method='singel', **kwargs):
        if method == 'group':
            fig, ax = plt.subplots(figsize=(8, 5))
            for key, result in self.results.items():
                freq   = result.psd.frequency
                db_psd = result.psd.amplitude
                plt.semilogx(freq, db_psd, label=key)
            plt.title('Self Noise')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('PSD [V**2/Hz dB]')
            plt.legend()
            plt.grid()
            _finalise_figure(fig, **kwargs)
        if method == 'singel':
            for key, result in self.results.items():
                fig, ax = plt.subplots(figsize=(8, 5))
                freq   = result.psd.frequency
                db_psd = result.psd.amplitude
                plt.semilogx(freq, db_psd, label=key)
                plt.title('Power Spectral Density')
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('PSD [V**2/Hz dB]')
                plt.legend()
                plt.grid()
                _finalise_figure(fig, **kwargs)
    
    ###########################################################################
    def freqresp_singelfreq(self, stream=None):
        if stream is not None:
            PreProcessStream(stream)
        for indx, inp in self.inputs.iterrows():
            if stream is None:
                stream = auto_read_data(inp, stream)
                PreProcessStream(stream)
            txt = ''.join([str(v).ljust(len(str(v))+4) for v in inp.values])
            print(txt)
            stime = utc(inp.starttime)
            etime = stime + 60 * inp.duration
            freq = inp.frequency
            st = stream.select(network=inp.network,
                               station=inp.station,
                               channel=inp.channel)
            st = st.slice(stime, etime)
            tr = st[0]
            # if input frequency is more than nyquist remove self noise of device
            #nyquist = (tr.stats.sampling_rate/2) - 0.1
            #if freq > nyquist:
            #    tr.filter('highpass', freq=nyquist)
            rms = _RMS(tr.data)
            ampl = rms * np.sqrt(2) / inp.gain
            ampl = 20 * np.log10(ampl)                                           # mutiple 10 convert to 20 (1400-03-09)
            key = f'{tr.id}-{int(inp.sampling_rate)}sps'
            if key not in self.results.keys():
                self.results[key] = result(key)
            self.results[key].freqresp_singelfreq.frequency.append(freq)
            self.results[key].freqresp_singelfreq.amplitude.append(ampl)

    ###########################################################################
    def freqresp_singelfreq_fft_plot(self, stream=None, savedir='fft-of-singleferq', **kwargs):
        # for all frequency
        if kwargs.get('save', False):                                          # Added 1400-03-09
            os.makedirs(savedir, exist_ok=True)                                # Added 1400-03-09
        fig1 = plt.figure()
        ax1 = plt.gca()
        colormap = plt.cm.gist_ncar
        num_plots = len(self.inputs)
        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
        if stream is not None:
            PreProcessStream(stream)
        for indx, inp in self.inputs.sort_index(ascending=False).iterrows():
            if stream is None:
                stream = auto_read_data(inp, stream)
                PreProcessStream(stream)
            txt = ''.join([str(v).ljust(len(str(v))+4) for v in inp.values])
            print(txt)
            stime = utc(inp.starttime)
            etime = stime + 60 * inp.duration
            freq_input = inp.frequency
            if freq_input >= 1:
                freq_input = int(freq_input)
            st = stream.select(network=inp.network,
                               station=inp.station,
                               channel=inp.channel)
            st = st.slice(stime, etime)
            st.detrend('constant')
            tr = st[0]
            name = f'{tr.id}-{int(inp.sampling_rate)}sps\nInput frequency is {freq_input}'
            tr.data = tr.data / inp.gain
            tr.plot(outfile=f'./{savedir}/{freq_input}HZ-{inp.vpp}vpp-{tr.id}-trace.png')
            tr.plot(starttime=tr.stats.starttime,
                    endtime=tr.stats.starttime + (1/freq_input*5),
                    outfile=f'./{savedir}/{freq_input}HZ-{inp.vpp}vpp-{tr.id}-trace-5T.png')
            tr.taper(0.05)
            data = tr.data
            delta = tr.stats.delta
            npts = tr.stats.npts
            sps = tr.stats.sampling_rate
            '''
            freq, psd = signal.welch(x=data, fs=sps, nperseg=None)
            #psd = 10 * np.log10(psd)
            ampl = psd
            '''
            segment = scipy.fftpack.helper.next_fast_len(npts)
            freq = np.fft.fftfreq(segment, d=delta)[:npts//2]
            ampl = scipy.fftpack.fft(data, segment) * delta
            ampl = np.abs(ampl[:npts//2]) / (segment*delta) # time of data = segment * delta
            print('max: ',max(ampl))
            ampl = 20 * np.log10(ampl)                                           # mutiple 10 convert to 20 (1400-03-09)
            # for a singel frequency figure
            
            fig2 = plt.figure()
            ax2 = plt.gca()
            ax2.semilogx(freq, ampl, label=f'{freq_input}HZ-{inp.vpp}vpp')
            ax2.legend(title='Input frequency')
            ax2.set_title('Recorded frequency')
            ax2.set_xlabel('Frequency [HZ]')
            ax2.set_ylabel('Amplitude [V db]')
            ax2.grid(which="both", ls='-.')                                      #  add 1400-03-09
            ax2.grid(which="major", ls='-')                                      #  add 1400-03-09
            savefile = f'./{savedir}/{tr.id}_{freq_input}HZ-{inp.vpp}vpp.png'
            _finalise_figure(fig2, savefile=savefile, **kwargs)
            
            # for all singel frequency figure
            ax1.semilogx(freq, ampl, label=f'{freq_input}HZ-{inp.vpp}vpp')
        ax1.set_title('Recorded frequency')
        ax1.set_xlabel('Frequency [HZ]')
        ax1.set_ylabel('Amplitude [V db]')
        ax1.legend(ncol=1, bbox_to_anchor=(1, 1), title='Input frequency')
        plt.grid(which="both", ls="-")                                          #  add 1400-03-09
        savefile = f'./{savedir}/{tr.id}_allHZ.png'
        _finalise_figure(fig1, savefile=savefile, **kwargs)

    ###########################################################################
    def freqresp_whitenoise(self, stream=None):
        for indx, inp in self.inputs.iterrows():
            print(inp)
            if stream is None:
                stream = auto_read_data(inp, stream)
            stime = utc(inp.starttime)
            etime = utc(inp.endtime)
            st = stream.select(network=inp.network,
                               station=inp.station,
                               channel=inp.channel)
            st = st.slice(stime, etime)
            PreProcessStream(st)
            st.plot(method='full')
            tr = st[0]
            data = tr.data / inp.gain
            delta = tr.stats.delta
            npts = tr.stats.npts
            segment = scipy.fftpack.helper.next_fast_len(npts)
            freq = np.fft.fftfreq(segment, d=delta)[:npts//2]
            ampl = scipy.fftpack.fft(data, segment)
            ampl = np.abs(ampl[:npts//2])
            ampl = 20 * np.log10(ampl)                                           # mutiple 10 convert to 20 (1400-03-09)
            key = f'{tr.id}-{int(inp.sampling_rate)}sps'
            if key not in self.results.keys():
                self.results[key] = result(key)
            self.results[key].freqresp_whitenoise.frequency = freq
            self.results[key].freqresp_whitenoise.amplitude = ampl

    ###########################################################################
    def plot_compare_freqresp(self, method='singelfreq', **kwargs):
        if method=='singelfreq':
            fig, ax = plt.subplots(figsize=(8, 5))
            for key, result in self.results.items():
                freq = result.freqresp_singelfreq.frequency
                ampl = result.freqresp_singelfreq.amplitude
                plt.semilogx(freq, ampl, label=key)
            plt.legend()
            plt.title('Frequency Response', fontweight='black')
            plt.xlabel('Frequency [Hz]', fontweight='black')
            plt.ylabel('Amplitude [V dB]', fontweight='black')
            plt.grid()
            _finalise_figure(fig, **kwargs)
        if method=='whitenoise':
            fig, ax = plt.subplots(figsize=(8, 5))
            for key, result in self.results.items():
                print(key)
                freq = result.freqresp_whitenoise.frequency
                ampl = result.freqresp_whitenoise.amplitude
                plt.semilogx(freq, ampl, label=key)
            plt.legend()
            plt.title('Frequency Response', fontweight='black')
            plt.xlabel('Frequency [Hz]', fontweight='black')
            plt.ylabel('Amplitude [V dB]', fontweight='black')
            plt.grid()
            _finalise_figure(fig, **kwargs)
        if method=='singelfreq VS whitenoise':
            for key, result in self.results.items():
                fig, ax = plt.subplots(figsize=(8, 5))
                #
                freq_whitenoise = result.freqresp_whitenoise.frequency
                ampl_whitenoise = result.freqresp_whitenoise.amplitude
                #norm_whitenoise = ampl_whitenoise[find_near_value_index(array=freq_whitenoise, value=80)]
                #ampl_whitenoise = ampl_whitenoise / norm_whitenoise
                #
                freq_singelfreq = result.freqresp_singelfreq.frequency
                ampl_singelfreq = result.freqresp_singelfreq.amplitude
                #norm_singelfreq = ampl_singelfreq[find_near_value_index(array=freq_singelfreq, value=80)]
                #ampl_singelfreq = ampl_singelfreq / norm_singelfreq
                #
                #max_passband_ampl_whitenoise = max(
                #        ampl_whitenoise[len(ampl_whitenoise)//2:])
                #ampl_whitenoise = ampl_whitenoise / max_passband_ampl_whitenoise
                plt.semilogx(freq_whitenoise, ampl_whitenoise,
                             label=key+' white-noise')
                #
                #ampl_singelfreq = np.array(ampl_singelfreq) / max(ampl_singelfreq)
                plt.semilogx(freq_singelfreq, ampl_singelfreq,
                             label=key+' singel-freq')
                title = 'Frequency Response\nWhite noise VS Singel freqency'
                plt.title(title, fontweight='black')
                plt.xlabel('Frequency [Hz]', fontweight='black')
                plt.ylabel('Amplitude [V db]', fontweight='black')
                plt.legend()
                plt.grid()
                _finalise_figure(fig, **kwargs)
    
    ###########################################################################
    def dynamic_range(self, stream=None, savepath='./'):
        # network, station, channel, starttime, duration, bit
        # how long data get to proccessing in minute
        database = []
        for indx, inp in self.inputs.iterrows():
            print(inp)
            if stream is None:
                stream = auto_read_data(inp, stream)
            amp_max = 2.0 ** (inp.bit - 1)
            starttime = utc(inp.starttime)
            st = stream.select(network=inp.network,
                               station=inp.station,
                               channel=inp.channel)
            st = st.slice(starttime, starttime+60*inp.duration)
            PreProcessStream(st)
            tr = st[0]
            # calculate mean of data
            mean = tr.data.mean()
            mean = round(mean, 2)
            #tr.plot()
            rms_n_detrend = _RMS(tr.data)
            rms_n_detrend = round(rms_n_detrend, 2)
            dynamicrange_n_detrend = 20 * np.log10(amp_max/rms_n_detrend)
            dynamicrange_n_detrend = round(dynamicrange_n_detrend, 2)
            tr.detrend('constant')
            tr.plot()
            rms_y_detrend = _RMS(tr.data)
            rms_y_detrend = round(rms_y_detrend, 2)
            dynamicrange_y_detrend = 20 * np.log10((amp_max-mean)/rms_y_detrend)
            dynamicrange_y_detrend = round(dynamicrange_y_detrend, 2)
            key = f'{tr.id}-{int(inp.sampling_rate)}sps'
            if key not in self.results.keys():
                self.results[key] = result(key)
            self.results[key].dynamic_range.sampling_rate = \
            tr.stats.sampling_rate
            self.results[key].dynamic_range.mean = mean
            self.results[key].dynamic_range.rms_n_detrend = rms_n_detrend
            self.results[key].dynamic_range.rms_y_detrend = rms_y_detrend
            self.results[key].dynamic_range.dynamicrange_n_detrend = \
            dynamicrange_n_detrend     
            self.results[key].dynamic_range.dynamicrange_y_detrend = \
            dynamicrange_y_detrend
            self.results[key].dynamic_range.id = tr.id
            database.append([tr.id, tr.stats.sampling_rate, mean,
                             rms_n_detrend, dynamicrange_n_detrend,
                             rms_y_detrend, dynamicrange_y_detrend])
        database = pd.DataFrame(np.array(database),
                                columns=['ID', 'SPS', 'Mean',
                                         'RMS(NoDetrend)',
                                         'DynamicRange(NoDetrend)',
                                         'RMS(Detrend)',
                                         'DynamicRange(Detrend)'
                                         ])
        self.dynamic_range_table = database
        savepath = os.path.join(savepath, 'Dynamic-Range.xlsx')
        database.to_excel(savepath, index=False)
        
    ###########################################################################
    def time_accuracy(self, stream=None, show=False, mean=None):
        '''
        network, station, channel, starttime, endtime
        '''
        for indx, inp in self.inputs.iterrows():
            if stream is None:
                stream = auto_read_data(inp, stream)
            st = stream.select(network=inp.network,
                               station=inp.station,
                               channel=inp.channel)
            st = st.slice(utc(inp.starttime), utc(inp.endtime))
            st.print_gaps()
            st.merge()
            tr = st[0]
            Length = round((tr.stats.npts * tr.stats.delta)/3600, 2)
            print(f'Length of trace is {Length:.3} hr')
            n = int(input('how many section do you need to analys? '))
            st = tr/n
            print(f'{len(st)} parts. Each part is {Length/n*60:.3} min.')
            time_errors_all_part = []
            key = f'{tr.id}-{int(inp.sampling_rate)}sps'
            if key not in self.results.keys():
                self.results[key] = result(key)
            self.results[key].time_error.starttime = tr.stats.starttime
            self.results[key].time_error.endtime = tr.stats.endtime
            for num, tr in enumerate(st):
                print(num)
                time_errors = _get_time_error(tr, mean=mean, show=show)
                time_errors_all_part = np.append(time_errors_all_part,
                                                 time_errors)
                counts, bins = np.histogram(time_errors, bins=30)
                mode_error = bins[counts.argmax()]
                self.results[key].time_error.trend.append(mode_error)
                self.results[key].time_error.parts[f'part{num}'] = \
                TimeError_part(counts, bins, time_errors,
                               tr.stats.starttime, tr.stats.endtime)
                self.results[key].time_error.all_errors = time_errors_all_part
                self.results[key].time_error.len_part = Length/n*60
    
    ###########################################################################
    def plot_timeerror_trend(self, **kwargs):
        for key, val in self.results.items():
            stime = val.time_error.starttime.strftime('%Y-%m-%d %H:%M:%S')
            etime = val.time_error.endtime.strftime('%Y-%m-%d %H:%M:%S')
            trend = val.time_error.trend
            len_part = val.time_error.len_part
            fig = plt.figure(figsize=(8, 6))
            plt.plot(trend, label=key)
            xlabel = f'Sections (each section is {len_part:.3} min)'
            plt.xlabel(xlabel, fontweight='black')
            plt.ylabel('TimeError (MODE)', fontweight='black')
            plt.legend()
            plt.grid()
            plt.title(f'Device: {key}\n' +
                      f'{stime} to {etime}', fontweight='black')
            #kwargs['savefile'] = f'{key}-trend.png'
            _finalise_figure(fig, **kwargs)
    
    ###########################################################################
    def plot_timeerror_hist(self, **kwargs):
        for key, val in self.results.items():
            for partname, part in val.time_error.parts.items():
                counts = part.counts 
                bins = part.bins
                #
                date = part.starttime.strftime('%Y-%m-%d')
                stime = part.starttime.strftime('%H:%M:%S')
                etime = part.endtime.strftime('%H:%M:%S')
                fig = plt.figure(figsize=(8, 6))
                plt.hist(bins[:-1], bins, weights=counts)
                plt.title(f'Device: {key}\n' +
                          f'{date} ({stime} to {etime})', fontweight='black')
                plt.xticks(rotation=30)
                plt.xlabel('TimeError', fontweight='black')
                plt.ylabel('Abundance [count]', fontweight='black')
                kwargs['savefile'] = f'{key}-{partname}.png'
                _finalise_figure(fig, **kwargs)
    
    ###########################################################################
    def PDF(self, stream=None, inventory=None, ppsd_length=3600.0, paz={
            'poles': [], 'zeros': [], 'gain': 600000.0,
            'sensitivity': 1.0}):
        from obspy.signal import PPSD
        from obspy.imaging.cm import pqlx
        for indx, inp in self.inputs.iterrows():
            if stream is None:
                stream = auto_read_data(inp, stream)
            st = stream.select(network=inp.network,
                               station=inp.station,
                               channel=inp.channel)
            st = st.slice(utc(inp.starttime),
                          utc(inp.starttime) + 60*inp.duration)
            PreProcessStream(st)
            tr = st[0]
            metadata = inventory or paz
            ppsd = PPSD(tr.stats, metadata=metadata, ppsd_length=ppsd_length)
            print('ppsd is here:', ppsd)
            ppsd.add(tr)
            name = f'{tr.id}-{tr.stats.sampling_rate}sps'
            ppsd.plot(f'{name}-ppsd.png', cmap=pqlx)
            ppsd.save_npz(f"{name}-ppsd.npz") 
    
    ###########################################################################
    def gain(self, stream=None, plotmethod='semilogx', outpath='None', **kwargs):
        database = []
        for indx, inp in self.inputs.iterrows():
            if stream is None:
                stream = auto_read_data(inp, stream)
            st = stream.select(network=inp.network,
                               station=inp.station,
                               channel=inp.channel)
            st = st.slice(utc(inp.starttime),
                          utc(inp.starttime) + 60*inp.duration)
            PreProcessStream(st)
            tr = st[0]
            # get rms amplitude
            rms = _RMS(tr.data)
            gain = (rms*np.sqrt(2)) / (inp.vpp/2)
            gain = round(gain, 2)
            database.append([tr.id, tr.stats.sampling_rate, inp.frequency,
                             inp.vpp, rms, gain])
        database = pd.DataFrame(np.array(database),
                                columns=['ID', 'SPS', 'frequency', 'Vpp',
                                         'RMS', 'Gain'])
        self.gain_table = database
        database.to_excel(f'{outpath}/Gain.xlsx', index=False)
        freqs = [float(_) for _ in database.frequency.values]
        gains = [float(_) for _ in database.Gain.values]
        savefile=os.path.join(outpath, 'Gain.png')                             # add in 1400-03-09
        fig = plt.figure()
        if plotmethod == 'liner':
            plt.plot(freqs, gains)
        elif plotmethod == 'semilogx':
            plt.semilogx(freqs, gains)
        plt.grid()
        _finalise_figure(fig, savefile=savefile,**kwargs)
    
    ###########################################################################
    def gain2(self, stream=None, **kwargs):
        database = []
        for indx, inp in self.inputs.iterrows():
            if stream is None:
                stream = auto_read_data(inp, stream)
            st = stream.select(network=inp.network,
                               station=inp.station,
                               channel=inp.channel)
            st = st.slice(utc(inp.starttime),
                          utc(inp.starttime) + 60*inp.duration)
            PreProcessStream(st)
            tr = st[0]
            # get rms amplitude
            rms = _RMS(tr.data)
            amp = rms * np.sqrt(2)
            gain = amp / (inp.vpp/2)
            gain = round(gain, 2)
            database.append([tr.id, tr.stats.sampling_rate, inp.frequency,
                             inp.vpp, rms, amp, gain])
        database = pd.DataFrame(np.array(database),
                                columns=['ID', 'SPS', 'frequency', 'Vpp',
                                         'RMS', 'Amplitude', 'Gain'])
        self.gain_table = database
        database.to_csv('Gain.xlsx', index=False)
        volts = [float(_)/2 for _ in database.Vpp.values]
        gains = [float(_)   for _ in database.Gain.values]
        ampls = [float(_)   for _ in database.Amplitude.values]
        fig = plt.figure()
        plt.plot(volts, ampls)
        plt.xlabel('Volt')
        plt.ylabel('Count')
        plt.grid()
        
        fig2 = plt.figure()
        plt.plot(volts, gains)
        plt.xlabel('Volt')
        plt.ylabel('Gain [count/volt]')
        plt.grid()
        _finalise_figure(fig, **kwargs)

def storage_test(inputfile):
    from obspy import read
    import os
    st = read(inputfile)
    st.write('STEIM1.msd', format='MSEED', encoding='STEIM1')
    st.write('STEIM2.msd', format='MSEED', encoding='STEIM2')
    #
    V0 = round(os.stat(inputfile).st_size/1e6, 2)
    V1 = round(os.stat('STEIM1.msd').st_size/1e6, 2)
    V2 = round(os.stat('STEIM2.msd').st_size/1e6, 2)
    print(f'Origin: {V0}\nSTEIM1: {V1}MB\nSTEIM2: {V2}MB')
    a = round(V0/V1, 2)
    print(f'Origin is {a} times more than STEIM1 standard')
    b = round(V0/V2, 2)
    print(f'Origin is {b} times more than STEIM2 standard')
    os.remove('STEIM1.msd')
    os.remove('STEIM2.msd')

