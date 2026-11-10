# Model terms — v1 and v2 weights

The root MIT license applies to the project code, **not** to the bundled weights.
The v1 and v2 weights were trained with the Namine Ritsu singing database:
**波音リツ / カノン** (Namine Ritsu; sung and provided by Canon).

Use and redistribution of these weights must comply with the database terms.
This document summarizes the checked Japanese text; it does not replace it or grant additional rights.

- [Official voicebank page](https://www.canon-voice.com/voicebanks/)
- [Official singing database folder](https://drive.google.com/drive/folders/1XA2cm3UyRpAk_BJb1LTytOWrhjsZKbSN)
- [Database-specific terms, full Japanese text](https://drive.google.com/file/d/1TgUg5hk9rDf7MXdNFnBoVwOMcWzr3At-/view)
- [General terms](https://www.canon-voice.com/terms/) explain that bundled database terms take precedence.

Checked/retrieved: **2026-11-10**, unnumbered Japanese database-terms version.
SHA-256 of the retrieved original: `5441c81f142584239fa9f589122570226adcc22ce15242b78af839fb6ce0315e`.

Article 7(1) permits commercial and noncommercial use under its conditions.
Article 7(4) requires both **波音リツ** and **カノン** in the credits of released software and AI models, even when only some information is used or databases are mixed.
Suggested credit: 歌声データベース: 波音リツ（歌唱・提供: カノン）.

Article 7(2) generally prohibits use that does not reflect the database voice, including serving as a base model for another database, transfer learning and fine tuning. Its specific exception concerns software producers whose software officially releases a model using this database voice, and requires a README disclosure. This is not blanket permission for users to adapt these weights to other voices. Article 7(3) additionally requires the exact prescribed text/ASCII art in the resulting model's `readme.txt` when its special conditions apply; consult the original rather than substituting an ordinary credit.

Article 9(3) applies the database terms to models generated from the database, including the conditions applicable to derived models. Keep these terms and credits with redistributed weights; the repository MIT license does not override them. This statement does not assert that every generated waveform becomes an MIT-licensed or identically licensed model.

The original recordings, labels, scores, MIDI and database lyrics are not included. Article 3 restricts republication of the original data; Article 9(4) also restricts reuse of the database songs/MIDI for other databases. Commercial-use permission is not unrestricted redistribution permission for the original database.

Article 9(1) allows terms to change without advance notice; Article 9(2) requires compliance with the provider's requests to stop publication/remove works. Consult the current official text. Questions of interpretation are reserved to the provider under Article 8. No separate waiver or additional permission has been obtained. Code or physics-based modeling does not exempt a trained model from these conditions. Voice-quality acceptance is not claimed.

日本語: コードはMIT、v1/v2の重みは波音リツ歌声DBの規約に従います。「波音リツ」「カノン」の両名の表示、他の声の基礎モデル等への利用制限と限定的な例外、特別な表示、派生モデルへの適用、改定・削除要請の条件を原文で確認してください。

## Other voices / 他の声

If you train a model on another corpus with the included code, the trained weights are governed by that corpus's terms, not by MIT. Check whether the corpus permits model training and model release, required credits, and prohibited uses before publishing. Never initialize another voice from the Namine Ritsu weights.

同梱のコードで他のコーパスから学習したモデルの重みは、MIT ではなく、そのコーパスの規約に従います。学習とモデルの公開が許されているか、クレジットの表記、禁止されている用途を確認してから公開してください。波音リツの重みを他の声の初期値にしないでください。
