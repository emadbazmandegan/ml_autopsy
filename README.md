# 🧠 Model Autopsy
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mlautopsy-grsr9d7hpt3vbetkgbs75j.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why I Built This

I got tired of flying blind.

You know the feeling. You train a model, you see `Accuracy: 0.94`, and you high-five your team. You deploy it. Two days later, a PM comes to your desk asking why the model is flagging every VIP user as "Fraud."

You spend the next three days writing ad-hoc Pandas scripts, grepping through logs, and manually staring at CSV rows to figure out what happened. You realize your model has a 40% error rate on users from New York who signed up on a Sunday.

The global metric lied to you.

I built **Model Autopsy** because I wanted to stop writing those throwaway debugging scripts. I wanted a tool that I could throw *any* model output at and immediately get back a report that says: "Here is exactly where you are screwing up."

It’s not magic. It’s just the automated surgical tools I wish I had five years ago.

---

## What It Actually Does

This isn't some black-box "Explainable AI" magic. It's a set of hard-nosed diagnostic tools that look for patterns in your failures.

### 1. It Finds the "Slices" of Death
Instead of you guessing "Maybe it's age? Maybe it's location?", the tool brute-forces it. It scans thousands of combinations of your features to find the specific subgroups where your model is failing.
> *Example: "Your model is great, except for 'Income < 50k' AND 'Education = Masters', where it fails 60% of the time."*

### 2. It Clusters Your Mistakes
Errors rarely happen randomly. usually, there's a specific *type* of user or input that trips up your model. This tool essentially runs a clustering algorithm on just your failed predictions to group them into "Failure Profiles."
> *Example: "Cluster 1 (30% of errors) are all users with missing phone numbers."*

### 3. It Stops You From Regressing
When you train a V2 model, the AUC might go up, but that doesn't mean it's better. V2 might fix 100 bugs from V1 but introduce 50 new ones that are way worse. The **Comparison Mode** highlights exactly which samples flipped from "Correct" to "Incorrect" so you don't accidentally ship a regression.

---

## How to Use It

I made this super simple because I hate configuring YAML files. You just need two CSVs: your **Dataset** (features) and your **Predictions** (what the model said).

### The "I just want to click buttons" way
[**👉 Try the Live Demo**](https://mlautopsy-grsr9d7hpt3vbetkgbs75j.streamlit.app/) (No installation required)

Or run it locally on your machine. No data leaves your laptop.

```bash
git clone https://github.com/emadbazmandegan/ml_autopsy.git
cd ml_autopsy
pip install -r requirements.txt

# Launches the web interface
streamlit run ui/app.py
```

### The "I have a CI pipeline" way
I also built a CLI so you can run this as a check in your build process.

```bash
pip install -e .

# Runs the autopsy and spits out a markdown report
ml-autopsy audit -d data.csv -p preds.csv -o report/
```

### "My columns are named weirdly"
That's fine. I added a bunch of heuristics to figure it out. If your ID column is named `customer_id` or `ImageID` or `foo`, it'll probably find it. If your target is `ground_truth` or `y_true`, it'll find it. You don't need to rename your columns to fit my tool.

---

## Under the Hood

This is built on the stack you already know: **Python, Pandas, Scikit-Learn, and Streamlit**.

There are about **150+ tests** covering the logic because I don't trust code that isn't tested, especially code that is supposed to be checking *other* code.

If you find a bug, open an issue. If you want to add a new way to find errors, send a PR. I'm actively using this for my own projects, so I'll probably see it.

---

**Stop shipping black boxes.**


---

## 🤝 Join the Surgical Team
This is an open-source movement to bring transparency to AI. We welcome:
- 🐛 Bug reports & diagnostic improvements
- 🧪 New slice discovery algorithms
- 🎨 UI/UX enhancements

Check out our [Contribution Guidelines](CONTRIBUTING.md) to get started.

---

**Don't just retrain. Understand.**  
Built with ❤️ for the reliable AI community.
