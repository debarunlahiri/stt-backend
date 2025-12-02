# Fix spaCy Initialization Error

## Problem

Argos Translate depends on spaCy, and spaCy may try to initialize or download models when imported, causing errors in offline environments.

## Solution

spaCy is a dependency of Argos Translate but is only used internally. We can suppress spaCy initialization errors by configuring it to work offline.

### Option 1: Disable spaCy Model Downloads (Recommended)

Set environment variable to prevent spaCy from trying to download models:

```bash
export SPACY_DISABLE_MODEL_DOWNLOAD=1
```

Add to your `.env` file:
```env
SPACY_DISABLE_MODEL_DOWNLOAD=1
```

Or add to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.):
```bash
echo 'export SPACY_DISABLE_MODEL_DOWNLOAD=1' >> ~/.zshrc
source ~/.zshrc
```

### Option 2: Suppress spaCy Warnings

If the error is just a warning, you can suppress it by adding this to your code before importing argostranslate:

```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='spacy')
```

### Option 3: Install spaCy Models Offline (If Needed)

If Argos Translate specifically needs spaCy models, you may need to download them on a machine with internet and transfer them.

However, Argos Translate typically doesn't require specific spaCy language models for basic translation - it uses its own translation models.

## Verification

After applying the fix, restart your server and check if the error is resolved:

```bash
python run.py
```

If you still see errors, check the specific error message and we can address it further.

## Common spaCy Initialization Errors

1. **Model download errors** - Set `SPACY_DISABLE_MODEL_DOWNLOAD=1`
2. **Missing model files** - Usually not needed for Argos Translate
3. **Import errors** - Ensure spaCy is properly installed: `pip install spacy`

## Note

Argos Translate uses spaCy internally for some processing but doesn't require specific language models to be downloaded. The translation models (`.argosmodel` files) are separate and must be installed separately as described in `DOWNLOAD_TRANSLATION_MODELS.md`.

