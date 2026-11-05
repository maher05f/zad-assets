"""يقيس أقصى استهلاك للذاكرة (peak) لكل نسخة عند تشغيلها على ملف صوتي (Windows).

    <venv>/python scripts/measure_ram.py --wav test-audio/sample-15s.wav out/muaalem-int8.onnx out/muaalem-fp32.onnx

كل نموذج يُشغَّل في عملية مستقلة: تحميل الجلسة ثم تشغيل واحد. يُطبع:
  base  = ذاكرة العملية بعد استيراد onnxruntime وقبل تحميل النموذج
  peak  = أقصى Working Set (ذاكرة فعلية) و أقصى Private Bytes طوال العملية
"""
import argparse, ctypes, json, os, subprocess, sys, time
from ctypes import wintypes


class PMC(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


_k32 = ctypes.WinDLL('kernel32', use_last_error=True)
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
_k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL


def mem(handle=None):
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    h = handle if handle is not None else _k32.GetCurrentProcess()
    if not _k32.K32GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return pmc


def child(model, wav, threads):
    import numpy as np, soundfile as sf, onnxruntime as ort
    x = sf.read(wav, dtype='float32')[0][None]
    base = mem().WorkingSetSize
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    t = time.time()
    s = ort.InferenceSession(model, o, providers=['CPUExecutionProvider'])
    load = time.time() - t
    after_load = mem().WorkingSetSize
    t = time.time()
    s.run(None, {'waveform': x})
    run = time.time() - t
    p = mem()
    print(json.dumps({'baseMB': base / 2**20, 'afterLoadMB': after_load / 2**20,
                      'peakWorkingSetMB': p.PeakWorkingSetSize / 2**20, 'peakPrivateMB': p.PeakPagefileUsage / 2**20,
                      'loadS': load, 'runS': run, 'audioS': x.shape[1] / 16000}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--child')
    ap.add_argument('models', nargs='*')
    a = ap.parse_args()
    if a.child:
        return child(a.child, a.wav, a.threads)
    res = {}
    for m in a.models:
        out = subprocess.run([sys.executable, '-I', '-X', 'utf8', os.path.abspath(__file__), '--child', m,
                              '--wav', a.wav, '--threads', str(a.threads)], capture_output=True, text=True, check=True)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        res[os.path.basename(m)] = r
        print('%-22s peak RAM %6.0f MB (private %6.0f MB) · قبل النموذج %4.0f MB · تحميل %.1fs · تشغيل %.1fs على %.1fs صوت'
              % (os.path.basename(m), r['peakWorkingSetMB'], r['peakPrivateMB'], r['baseMB'], r['loadS'], r['runS'], r['audioS']))
    print(json.dumps(res))


if __name__ == '__main__':
    main()
