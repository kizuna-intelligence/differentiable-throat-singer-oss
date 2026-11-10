# CHIMERA renderer provenance

Source: https://github.com/jamesondh/chimera-vocal-synth
Commit: `092b16ce12b927d473c884ca2dfae53ce1b995df`.
The copied rendering code is unmodified (byte-for-byte). The source files and original SHA-256 values are listed below. Only the playback connections are authored here: `vite.config.ts` replaces React/audio/store bindings with replay bindings and extracts TractMap from engine.ts, erasing type annotations. This is a saved-state playback adapter, not the complete upstream app or WASM engine. The video credit is drawn by our compositor, outside the original renderer.
`WTFPL.txt` is an added license text obtained from https://github.com/scruss/AVHershey-OTF/blob/master/COPYING ; it is not represented as an original CHIMERA file.

| Original path | bytes | Original SHA-256 |
|---|---:|---|
| `LICENSE` | 1070 | `0b8d6e0649e7b2068a8300cafb9ebded80d9a0fe99e699b96538ba0a8c5236b1` |
| `public/fonts/AVHERSHEY-LICENSE.txt` | 380 | `d27b30a1f3903dbecc7fa72cf507459ac2638049f987a9bde857df5fa9e4ba08` |
| `public/fonts/AVHersheySimplexLight.otf` | 11540 | `e20a7cbafa2f84cb21cbeec109b488d00c9bb588f0e526011dd0a35336664cfd` |
| `public/fonts/AVHersheySimplexMedium.otf` | 13564 | `919ae9cf082647290104861d305390268b55933c9fd8929fb852143377fbf7ea` |
| `public/fonts/OFL.txt` | 4388 | `108b4e57c9c796d3d38d0428ca7ee39de47ad93187302718d9b2d8864b9b716b` |
| `public/fonts/archivo-latin-ext.woff2` | 86240 | `5717f37059660ca5c899bad6c48ee22c3ac55cb3c484055241689d0f905a1a86` |
| `public/fonts/archivo-latin.woff2` | 90104 | `e3a28eade21a900c7155a247757f4b2834c07bb7ef07ad7efa55cebaac1e8f5e` |
| `src/ui/analysis.ts` | 5510 | `025125cb1d69d1980a5b10a2af23a45d4a8809ba7a6728f329945d7db4bfcdba` |
| `src/ui/body.ts` | 2530 | `eff9f2d9e1680b521b025e7d57a035d3b4d0854c94e04a28502a9dc74db8f9d3` |
| `src/ui/colors.ts` | 3469 | `628ca90657dc70b452f521085afe94a8c20cc67fcf3189111a2503fe82b322da` |
| `src/ui/components/Stroboscope.tsx` | 4837 | `500d123ac3a739e00c1397a3b4652551bedcb18f8e5c3df891af1e6ccb0a88b6` |
| `src/ui/components/TubeView.tsx` | 4469 | `1e1ce551eb3f30b99d6827a48bd10b1b6df4a0986107de016e9c72b8a8ad2bd3` |
| `src/ui/render/anatomy.ts` | 20688 | `b95baba9a50ecd275fd16a63cd4aca03cc37ff93b3b5556e28e6feb84fd4ad12` |
| `src/ui/render/expression.ts` | 6386 | `7ebad4e502b1e1e088ff247518873687b21c89db7d347772e4cb6498ec7daafe` |
| `src/ui/render/geom.ts` | 6495 | `5d2b2b4026599db13e6a9d19be43c46c3411aab4bd9f02ad138afe77a5b01d5b` |
| `src/ui/render/profile.ts` | 9774 | `b880cc058c51b10ffad895f36c19ca57686da8b94e5d8cd48009c83a9e9c23d7` |
| `src/ui/render/readouts.ts` | 4224 | `790d8a0fc785e31658c7673d386d062c40142821dc95db785a43417223c79ee5` |
| `src/ui/render/vector.ts` | 13577 | `e90a3c321d4ec4bebbfe0b5e8b760231da39cb295aba2848b43154988810d550` |
| `src/ui/styles.css` | 52500 | `dbfbc752a2653a44210b1566b4f99a14576b1dc73946b7cbc65e3165a16837f4` |
| `src/wasm/engine.ts` | 10533 | `ccd61bdbece2bdf8975ce28b62a487ca0b40d5639d71f3b11201c058d013b5b7` |
| `src/wasm/types.ts` | 6840 | `a9d6b78c8a3693e71a9a317fd616f8f983279a15f1354206606bde206e120486` |
| `src/worklet/protocol.ts` | 2592 | `23fbafc2124d1f512b3d9cf19773e0cba24148e258b8a4c5bad0de3ebc3e0ffb` |
