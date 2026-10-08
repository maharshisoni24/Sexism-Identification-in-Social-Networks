# A Beginner's Guide to Our ML Sexism Detection Project

Welcome! If you are reading this, you are about to learn how we built a machine learning system to detect sexism on Twitter. You don't need any prior ML experience to understand this guide. We will use simple analogies to explain everything from the ground up, so you can confidently present this project and answer questions.

---

## Part 1: Background Concepts

Before we dive into our specific project, let's cover the basics.

*   **What is ML (Machine Learning)?** Imagine trying to teach a child to recognize a cat. You don't give them a list of rules ("has pointy ears, whiskers, meows"). Instead, you show them hundreds of pictures of cats and dogs, telling them which is which. Eventually, they learn the underlying patterns. Machine learning does the same thing with computers: it learns patterns from data instead of being explicitly programmed.
*   **What is NLP (Natural Language Processing)?** This is the subfield of AI focused on helping computers understand, interpret, and generate human language. It's how Siri understands you, or how Google translates text.
*   **What is Text Classification?** A common NLP task. It's like sorting your mail. You look at an envelope and classify it into a bin: "Bills," "Personal," or "Junk." We are classifying tweets into "Sexist" or "Not Sexist."
*   **What are Embeddings?** Computers only understand numbers, not words. Embeddings translate words (or whole sentences) into a long list of numbers. Think of it as a GPS coordinate for meaning. The sentence "I love dogs" and "Puppies are great" will have similar "coordinates" because their meanings are close, while "Taxes are due" will be far away.
*   **What is a Neural Network?** The "brain" of our system. It's a series of mathematical layers. Information (our numbers) goes in one end, gets multiplied and added together in complex ways across several layers, and a prediction comes out the other end.
*   **What is a Loss Function?** This is the teacher's grading rubric. When the neural network makes a prediction, the loss function compares it to the real answer and calculates a "loss" score. A high score means the network was very wrong. The network uses this score to adjust its math so it does better next time.
*   **What is Multi-Task Learning?** Imagine a student studying for a math test and a physics test at the same time. Learning one helps with the other because they share underlying concepts. We train our model to do two related tasks at once, hoping it makes the model smarter overall.
*   **What is ECE / Calibration?** If a weather app says there is a 70% chance of rain, and it says that 100 times, it should actually rain exactly 70 times. If it rains 90 times, the app is poorly calibrated. ECE (Expected Calibration Error) measures this. We want our model to be well-calibrated: if it's 80% confident a tweet is sexist, we want it to be right 80% of the time.

---

## Part 2: The Problem

### Online Sexism
Social media platforms are flooded with toxic content. We need automated systems to detect sexism because humans can't possibly read millions of tweets a day. However, it's a very difficult problem.

### Why is it hard?
It's not just a list of bad words. Sexism involves sarcasm, cultural context, subtle stereotypes, and borderline cases. For example, "Women are naturally better caregivers" is a stereotype, but a simple keyword filter won't catch it. 

### The Dataset: EXIST 2021
We use a dataset from a competition called EXIST 2021. It contains 6,977 tweets in both English and Spanish. Human annotators read these tweets and labeled them.
However, **Annotator Disagreement** is a huge issue. Because sexism can be subjective, the people labeling the data often disagreed with each other on the borderline cases!

### Two Tasks in One
We train our model to do two things simultaneously:
*   **Task 1 (Binary):** Is the tweet sexist or not? (Yes/No)
*   **Task 2.3 (Categorical):** If it is sexist, which of the 5 types is it? (e.g., stereotyping, sexual violence, objectification).

---

## Part 3: Our Baseline Model

Before getting fancy, we built a standard, solid foundation. 

1.  **Text Embeddings:** We use a tool called `paraphrase-multilingual-mpnet-base-v2`. It takes a tweet (English or Spanish) and turns it into a list of 768 numbers. Why 768? It's just the size of the "map" this specific tool uses to capture the nuances of meaning.
2.  **SwiGLU Blocks:** These are specialized layers in our neural network. Think of them as a set of gates that decide which parts of the meaning are most important for finding sexism, filtering out the noise.
3.  **Classification Heads:** The final layers. One head outputs a yes/no probability for Task 1. The other head outputs 5 probabilities for Task 2.
4.  **Kendall Multi-Task Loss:** Since we are doing two tasks, we need a way to balance them. This standard technique automatically decides how much attention the model should pay to Task 1 versus Task 2 during training.

### Baseline Problems
Our baseline is okay, but it suffers from common ML flaws:
*   **Equal Weighting:** It spends just as much time learning from obvious tweets as it does from the really hard, ambiguous ones.
*   **Inconsistency:** It can contradict itself (e.g., saying it's 10% likely to be sexist, but 80% likely to be 'stereotyping').
*   **Overconfidence:** It often guesses 100% yes or 100% no on tweets that humans argue about.

---

## Part 4: Our 3 Novelties

To fix the baseline's problems, we didn't build a bigger network. We changed the *Loss Function*—the grading rubric. We introduced three novelties (N1, N2, N3).

### N1 - Confidence-Aware Focal Loss
*   **The Intuition:** A good teacher spends more time helping the confused students than the straight-A students. N1 makes the model focus its learning power on the tweets it finds confusing.
*   **How it works:** 
    *   We calculate uncertainty: `d_i = 1 - |2p_i - 1|`. If the model's prediction (`p_i`) is 0.5 (a total guess), `d_i` is 1 (max uncertainty). If it predicts 0.99, `d_i` is near 0.
    *   We create a multiplier: `m_i = (1-α) + α·d_i^γ`. This number gets bigger when uncertainty is high.
    *   We multiply the standard loss by `m_i`. 
*   **The Result:** If the model is confused, the penalty for being wrong is massive. If it's confident and correct, the penalty is tiny. The model is forced to figure out the hard cases. (α=0.5, γ=2 are just tuning knobs we set).

### N2 - Hierarchical Consistency Regularization
*   **The Intuition:** You cannot logically say "This is NOT an animal" but also say "This is definitely a dog." N2 stops the model from making logical contradictions between Task 1 and Task 2.
*   **How it works:** We look at the probability for Task 1 (`p_sexist`) and the probability for a specific Task 2 category (`p_cat`).
    *   If `p_cat > p_sexist`, that's a contradiction!
    *   We apply a mathematical penalty: `Penalty = λ × max(0, p_cat - p_sexist)^2`. 
*   **The Result:** The model quickly learns the logical rule that the sub-category probability can never exceed the main category probability. (λ=0.5 is how hard we slap its wrist for failing).

### N3 - Adaptive Label Smoothing
*   **The Intuition:** For an obvious slur, the label is 100% sexist. But for a debatable, sarcastic joke, forcing the model to be 100% confident makes it arrogant and brittle. N3 tells the model, "For hard cases, it's okay to aim for 85% instead of 100%."
*   **How it works:** We use the uncertainty `d_i` from N1.
    *   Smoothing amount: `ε_i = ε_max × d_i`. The more uncertain the model is, the more we soften the target label. (ε_max=0.25 is the max softening allowed).
    *   We change the target label `y_i` to a softer version `ỹ_i`.
*   **The Result:** The model stops being overly confident on gray-area tweets, which makes it much more reliable in the real world.

---

## Part 5: Experiments & Results

How do we prove these ideas work? We use an **Ablation Study**.
An ablation study is like testing a recipe by leaving out ingredients one by one. We tested the baseline alone, then added N1, then N2, then N3, then pairs (N1+N2, etc.), and finally all three. This gives us 7 configurations to compare.

### Key Results
*   **Task 1 (Sexist or Not):** The combination of **N1+N3** gave us the best F1 score (a measure of accuracy for imbalanced data). It scored 0.7129, beating the baseline's 0.7002.
*   **Task 2 (Categories):** **N1 alone** worked best here, jumping to 0.6033 from the baseline's 0.5623.
*   **Per-Uncertainty-Bin Analysis:** We grouped tweets by how hard they were. The data proved that N1 improved scores *specifically* on the most difficult tweets, confirming our theory.
*   **Violation Stats:** We counted logical contradictions. The baseline failed logic 55.9% of the time. Adding N2 dropped that to 44.8%.
*   **Calibration:** N1+N2 gave the best ECE score (0.4064 vs baseline 0.4387), meaning the model's confidence is now much more trustworthy.

---

## Part 6: Q&A Preparation

If you present this, people will ask questions. Here is how to answer 20 likely questions simply.

**1. Why focus on the loss function instead of a bigger model?**
Because throwing computing power at a problem doesn't fix underlying logical flaws or handling of ambiguity. Smarter training is better than just bigger models.

**2. Why didn't you fine-tune an LLM like Gemini or BERT?**
We hit API limits with Gemini. Using pre-computed embeddings and a small neural network allowed us to do rapid, inexpensive research on our laptop.

**3. What does "embeddings: paraphrase-multilingual-mpnet-base-v2" mean?**
It's just the name of the specific tool we downloaded to turn text into numbers. We picked it because it understands both English and Spanish well.

**4. Why is the dataset only 6977 tweets? Isn't that small?**
Yes, but high-quality human labeling is expensive. That's exactly why techniques like N1, N2, and N3 are needed—to squeeze every drop of learning out of a small, difficult dataset.

**5. What is an F1 score?**
If a dataset is 90% "Not Sexist", a model that always guesses "Not Sexist" is 90% accurate, but totally useless. F1 score balances precision (not false alarms) and recall (catching the real stuff), giving a much fairer grade.

**6. Explain the math in N1 (Confidence-Aware Focal Loss) simply.**
It dynamically changes the penalty for being wrong. If the model is confident and right, penalty = 0. If it's guessing randomly (0.5), we multiply the penalty to say "Pay attention to this one!"

**7. How is N1 different from regular Focal Loss?**
Regular Focal Loss treats all hard examples the same. Ours looks at the model's *current* confusion level to dynamically adjust the weight on the fly.

**8. Explain N2 (Hierarchical Consistency) simply.**
It's a penalty for stupidity. If the model says a tweet has a 50% chance of being sexist, it physically cannot have an 80% chance of being "stereotyping." If it does that, N2 adds a math penalty to the loss score.

**9. Why doesn't N2 just force the numbers to match after the fact?**
Because we want the model to *learn* the relationship organically during training, making its internal representations smarter, rather than just slapping a band-aid on the final output.

**10. Explain N3 (Adaptive Label Smoothing) simply.**
Normally, we tell the model "This is 100% sexist." But if human annotators argued about it, it's really maybe 85% sexist. N3 softens the target so the model doesn't become overly stubborn and confident on gray areas.

**11. Why did N1+N3 work best for Task 1, but N1 alone for Task 2?**
Task 1 (yes/no) is very broad and subjective, so softening the labels (N3) helps it handle the gray area. Task 2 (specific categories) is more rigid; softening the labels might blur the lines between categories too much.

**12. What does ECE (Expected Calibration Error) actually mean in practice?**
It means if our system flags a tweet and says "I am 90% sure this is sexist," a human moderator can actually trust that 9 out of 10 times it will be right. The baseline model was arrogant and lied about its confidence.

**13. What is an Ablation Study?**
It's the scientific method of turning features on and off to prove what is actually causing the improvements, rather than just guessing.

**14. What are α=0.5, γ=2, λ=0.5?**
They are hyperparameters. Think of them as the volume knobs on our novelties. We tuned them through trial and error to find the sweet spot.

**15. Did you change the neural network architecture?**
Hardly at all. We used a standard setup (SwiGLU blocks). We wanted to prove our *loss functions* were the hero, not a crazy new network design.

**16. What's the biggest limitation of this work?**
It relies heavily on the quality of the initial embeddings. If the MPNet tool completely misunderstands a new slang word, our loss functions can't fully save it.

**17. What are SwiGLU blocks?**
A type of neural network layer that is very good at filtering information. It uses a "gate" to decide what data passes through to the next layer and what gets blocked.

**18. Why use both English and Spanish?**
The EXIST dataset provided both, and sexism is a cross-cultural problem. By using a multilingual embedding model, our system can learn patterns in one language and sometimes apply them to the other.

**19. What is Kendall Multi-Task Loss?**
When learning two tasks, one might be harder and dominate the training. Kendall loss is an automatic scale that balances the weights of Task 1 and Task 2 so they learn harmoniously.

**20. What is the main takeaway of this project?**
When dealing with subjective, ambiguous real-world data like sexism, standard models fail because they want black-and-white answers. By designing training rules that explicitly handle uncertainty and logic, we get better, more trustworthy results.
