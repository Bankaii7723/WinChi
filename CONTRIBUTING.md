# 🙌 Thank You!

Thank you for choosing to contribute to the **WinChi** project! We appreciate every bug report, feature suggestion, and pull request!

This guide will help you make your contributions clean, clear, and consistent with the rest of the codebase. Please take a moment to read through it before submitting anything.

---

## 🧠 Writing Code

When writing code for WinChi, please make sure to:

- **Comment and document clearly**: Future devs will thank you. Write as if someone else will read your code.
- **Check for bugs**: Test your code thoroughly.
- **Keep it clean**: Remove unused variables, dead code, and TODOs.

---

## 📁 Project Structure

Please get familiar with the general structure of the repository:

```
WinChi/
├─ .github/
├─ src/
|  ├─ config.ini
│  ├─ app.py
│  ├─ camel.py
│  ├─ assets/
│  │  ├─ icon.svg
|  ├─ requirements.txt
├─ .gitattributes
├─ .gitignore
├─ CODE_OF_CONDUCT.md
├─ CONTRIBUTING.md <-- you're reading this right now!
├─ LICENSE
├─ README.md
```


If you’re adding a new file or module, place it in the appropriate directory. If you’re not sure where it goes, ask us!

---

## 🔧 Setting Up Your Dev Environment

Here’s how to get started:

1. Clone the repo:
   ```bash
   git clone https://github.com/Bankaii7723/WinChi.git
   cd WinChi/src
   ```
2. Install dependencies:
    ```bash
    pip install requirements.txt
    ```
3. Run it locally:
    ```bash
    python app.py
    ```
> **Don't forget!** You need to import a ```.gguf``` model file when using the app! 
> **Extra tip!** WinChi is optimized for LlaMA 3 chat prompts. View ```camel.py``` for further optimizations.

# 🔁 Submitting a pull request

Before opening a PR:

- Make sure your branch is up to date with ```main```.

- Add a clear, descriptive PR title and summary.

# 💬 Got Questions?
Open an issue or start a discussion! We're here to help!

Thanks again, and happy contributing! 🛠️✨
