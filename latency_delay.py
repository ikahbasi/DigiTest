"""
Created on Tue May  4 19:23:24 2021

@author: imon
"""
from obspy import UTCDateTime as utc
class get_data:
    '''
    https://docs.gempa.de/seiscomp3/current/apps/scqc.html
    '''
    def __init__(self, savef='latency-delay.csv'):
        self.savef = savef
        self.lst = ["Latency[s]", "Delay[s]", "Packet-Size[s]", "Samples"]
        if savef:
            self.outfile = open(savef, 'w')
        self._printtxt()
    def _printtxt(self, sep=','):
        txt = sep.join(self.lst)
        if self.savef:
            self.outfile.write(txt)
        else:
            print(txt)
    def latency(self, trace):
        now = utc()
        self.latency   = now - trace.stats.starttime
        self.delay     = now - trace.stats.endtime
        self.lenpacket = trace.stats.endtime - trace.stats.starttime
        self.samples_per_packet = trace.stats.npts
        self.lst = [self.latency,
                    self.delay,
                    self.lenpacket,
                    self.samples_per_packet]
        self.lst = [str(round(_, 2)) for _ in self.lst]
        self._printtxt()