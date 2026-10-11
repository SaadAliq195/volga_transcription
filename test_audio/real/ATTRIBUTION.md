# Test audio: sources and licences

These clips are unmodified excerpts from open speech datasets, used only to test
the pipeline. They keep their original licences (below); the reference
transcripts in `manifest.json` come from the same datasets. They were fetched
with `scripts/fetch_test_audio.py` through the Hugging Face dataset viewer.

## LibriSpeech ASR corpus

- Licence: CC BY 4.0
- Source: https://www.openslr.org/12 (via `openslr/librispeech_asr` on Hugging Face)
- Citation: V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015
- 22 clips, 14 speakers:

  - `lsc_1284_0005.flac`: speaker LS-1284
  - `lsc_1284_0020.flac`: speaker LS-1284
  - `lsc_2094_0038.flac`: speaker LS-2094
  - `lsc_2094_0049.flac`: speaker LS-2094
  - `lsc_4446_0013.flac`: speaker LS-4446
  - `lsc_4446_0017.flac`: speaker LS-4446
  - `lsc_5639_0031.flac`: speaker LS-5639
  - `lsc_5639_0033.flac`: speaker LS-5639
  - `lsc_6930_0003.flac`: speaker LS-6930
  - `lsc_6930_0006.flac`: speaker LS-6930
  - `lsc_7127_0026.flac`: speaker LS-7127
  - `lsc_7127_0027.flac`: speaker LS-7127
  - `lsc_8230_0039.flac`: speaker LS-8230
  - `lsc_8230_0043.flac`: speaker LS-8230
  - `lsc_8455_0018.flac`: speaker LS-8455
  - `lsc_8455_0022.flac`: speaker LS-8455
  - `lso_3331_0014.flac`: speaker LS-3331
  - `lso_3528_0113.flac`: speaker LS-3528
  - `lso_4198_0024.flac`: speaker LS-4198
  - `lso_5764_0097.flac`: speaker LS-5764
  - `lso_6938_0018.flac`: speaker LS-6938
  - `lso_7902_0014.flac`: speaker LS-7902

## CSTR VCTK Corpus

- Licence: CC BY 4.0
- Source: https://datashare.ed.ac.uk/handle/10283/3443 (via `sanchit-gandhi/vctk` on Hugging Face)
- Citation: J. Yamagishi, C. Veaux, K. MacDonald, CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92), University of Edinburgh, 2019
- 12 clips, 12 speakers:

  - `vctk_p225_002.flac`: speaker VCTK-p225, female, English (Southern England)
  - `vctk_p237_304.flac`: speaker VCTK-p237, male, Scottish (Fife)
  - `vctk_p253_344.flac`: speaker VCTK-p253, female, Welsh (Cardiff)
  - `vctk_p264_337.flac`: speaker VCTK-p264, female, Scottish (West Lothian)
  - `vctk_p278_294.flac`: speaker VCTK-p278, male, English (Cheshire)
  - `vctk_p292_362.flac`: speaker VCTK-p292, male, NorthernIrish (Belfast)
  - `vctk_p297_207.flac`: speaker VCTK-p297, female, American (New York)
  - `vctk_p301_078.flac`: speaker VCTK-p301, female, American (North Carolina)
  - `vctk_p313_208.flac`: speaker VCTK-p313, female, Irish (County Down)
  - `vctk_p335_086.flac`: speaker VCTK-p335, female, NewZealand (English)
  - `vctk_p347_206.flac`: speaker VCTK-p347, male, SouthAfrican (Johannesburg)
  - `vctk_p376_245.flac`: speaker VCTK-p376, male, Indian

## Edinburgh International Accents of English Corpus (EdAcc)

- Licence: CC BY-SA 4.0
- Source: https://groups.inf.ed.ac.uk/edacc/ (via `edinburghcstr/edacc` on Hugging Face)
- Citation: R. Sanabria et al., "The Edinburgh International Accents of English Corpus", ICASSP 2023
- 6 clips, 6 speakers:

  - `edacc_EDACC-C05-A_5.wav`: speaker EDACC-C05-A, female, Romanian
  - `edacc_EDACC-C05-B_4.wav`: speaker EDACC-C05-B, female, Indian English
  - `edacc_EDACC-C08-A_0.wav`: speaker EDACC-C08-A, male, Scottish English
  - `edacc_EDACC-C10-B_1.wav`: speaker EDACC-C10-B, male, Don't know
  - `edacc_EDACC-C19-A_2.wav`: speaker EDACC-C19-A, male, Chinese
  - `edacc_EDACC-C21-B_3.wav`: speaker EDACC-C21-B, female, Spanish
