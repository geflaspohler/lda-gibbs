from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

#(rate, sig) = wav.read("sound/Sa146a001.wav")
(rate, sig) = wav.read("sound/english.wav")
mfcc_feat = mfcc(sig, rate)
fbank_feat = logfbank(sig, rate);

print(mfcc_feat)
