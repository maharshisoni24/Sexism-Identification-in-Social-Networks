# Slide 1: Title
**Say:** Hello everyone! Today we're presenting our project on detecting sexism in online text. We focused on the EXIST 2021 dataset and introduced three novel loss functions to improve our model's performance on tricky, ambiguous cases.
**Emphasize:** We didn't just try a new model architecture; we changed *how* the model learns from difficult examples.
**If asked:** EXIST 2021 is a dataset of English and Spanish tweets labeled for sexism.

# Slide 2: Agenda
**Say:** Here is what we'll cover today. We'll start with the problem we're trying to solve and the dataset we used. Then, we'll look at our baseline model and its limitations. After that, we'll introduce our three novelties, show you the design of our experiments, and finally, present our results.
**Emphasize:** We will clearly trace the path from the problems with standard models to our specific solutions.
**If asked:** Yes, we will cover both the theoretical intuition and the practical results.

# Slide 3: Problem
**Say:** Sexism on social media is a widespread problem that causes real harm. Automating its detection is crucial for moderation, but it's incredibly hard. People use sarcasm, cultural references, and subtle phrasing. Often, even humans disagree on whether a tweet is sexist or not.
**Emphasize:** Sexism detection is rarely black-and-white; it's full of gray areas.
**If asked:** We are focusing specifically on text-based sexism, not images or videos.

# Slide 4: Dataset
**Say:** We used the EXIST 2021 dataset. It contains nearly 7,000 training tweets in both English and Spanish. We tackle two tasks at once: Task 1 is a simple yes/no for sexism. Task 2.3 is more specific, asking which of 5 types of sexism is present, if any. 
**Emphasize:** We are predicting two things at once: "Is it sexist?" and "If so, what kind?"
**If asked:** The 5 types include things like stereotyping, objectification, and sexual violence.

# Slide 5: Baseline Architecture
**Say:** Let's look at our starting point, the baseline model. We take a tweet and use a pre-trained model to turn it into embeddings—basically a list of numbers representing its meaning. These numbers pass through neural network layers, which then branch out to make two predictions: one for Task 1 and one for Task 2.
**Emphasize:** This is a standard multi-task setup. Text goes in, two predictions come out.
**If asked:** We used the paraphrase-multilingual-mpnet-base-v2 model for embeddings because it handles both English and Spanish well.

# Slide 6: Baseline Limitations
**Say:** But this baseline has three major flaws. First, it treats easy and hard tweets equally, getting stuck on obvious cases instead of learning from the tricky ones. Second, it can contradict itself, like saying a tweet is *not* sexist, but also saying it contains *stereotyping*. Third, it's often too confident when it shouldn't be, blindly guessing 100% on ambiguous text.
**Emphasize:** The standard approach lacks nuance, consistency, and a sense of its own uncertainty.
**If asked:** These limitations are common in standard cross-entropy and multi-task learning setups.

# Slide 7: Our Approach
**Say:** To fix these issues, we designed three novel loss functions. Think of a loss function as the teacher's grading rubric. By changing the rubric, we change how the model learns. We call these N1, N2, and N3, and each specifically targets one of the baseline's limitations.
**Emphasize:** We are changing the *training signal*, not the neural network's architecture.
**If asked:** They can be combined or used separately, which we will show in our experiments.

# Slide 8: N1 (Confidence-Aware Focal Loss)
**Say:** Our first novelty, N1, is the Confidence-Aware Focal Loss. Simply put, if the model is really confused about a tweet (like predicting a 50/50 chance), N1 forces the model to pay more attention to it during training. It heavily penalizes mistakes on uncertain, hard examples, while largely ignoring the easy ones the model already understands.
**Emphasize:** N1 forces the model to focus its learning capacity on the hardest examples.
**If asked:** It's inspired by Focal Loss but specifically uses the model's own output probability to dynamically adjust weights.

# Slide 9: N2 (Hierarchical Consistency Regularization)
**Say:** Next is N2, which ensures logical consistency. Remember the contradiction issue? If the model predicts a 20% chance of being sexist, but a 60% chance of being a specific *type* of sexism, N2 slaps it with a penalty. It enforces the rule: the probability of a specific sub-type cannot exceed the probability of the main category.
**Emphasize:** N2 teaches the model basic logic: you can't have a sub-category without the parent category.
**If asked:** This is implemented as a penalty term added to the total loss, only active when a violation occurs.

# Slide 10: N3 (Adaptive Label Smoothing)
**Say:** Finally, N3 addresses overconfidence. Normally, labels are strictly 0 or 1. But for ambiguous tweets, forcing the model to predict exactly 1 or 0 makes it stubbornly overconfident. N3 looks at how uncertain the model is. If it's a hard case, we 'soften' the target label slightly, say to 0.85 instead of 1.0, teaching the model that it's okay to be slightly uncertain.
**Emphasize:** N3 prevents the model from being arrogantly wrong on debatable cases.
**If asked:** Standard label smoothing applies the same softening everywhere; ours adapts based on the difficulty of each tweet.

# Slide 11: Ablation Design
**Say:** To see if our ideas actually work, we did an ablation study. This means we tested every possible combination. We ran the baseline alone, then added just N1, just N2, just N3, all the pairs, and finally all three together. This proves exactly which feature is responsible for which improvement.
**Emphasize:** We tested 7 different configurations to isolate the impact of each novelty.
**If asked:** An ablation study is the gold standard for proving a new technique isn't just a lucky fluke.

# Slide 12: Results
**Say:** And the results were fantastic! For Task 1 (binary detection), combining N1 and N3 gave us the best F1 score of 0.7129, beating the baseline's 0.7002. For Task 2 (categorization), N1 alone performed best, jumping from 0.5623 to 0.6033. Our novelties objectively made the model more accurate.
**Emphasize:** The improvements are consistent across both tasks, proving our methods work.
**If asked:** F1 score is a balance of precision and recall, much better than plain accuracy for imbalanced data.

# Slide 13: Per-Uncertainty-Bin
**Say:** We also dug deeper to see *where* the model improved. We grouped the tweets by how hard they were. We found that our N1 loss significantly boosted performance specifically on the most difficult, ambiguous tweets—exactly as we designed it to do. It didn't just memorize the easy ones better.
**Emphasize:** We proved that our method solves the exact problem we targeted: the hard cases.
**If asked:** We measured this by looking at error rates across different ranges of predicted probabilities.

# Slide 14: Violation Stats
**Say:** What about N2 and logical consistency? We counted how often the model contradicted itself. The baseline model had logical violations on nearly 56% of predictions! By adding N2, we slashed that down to under 45%. Furthermore, N1+N2 gave us the best calibration score, meaning the model's confidence scores are much more trustworthy now.
**Emphasize:** N2 directly reduces embarrassing logical errors and makes the model's confidence more reliable.
**If asked:** Calibration is measured by Expected Calibration Error (ECE); a lower ECE is better.

# Slide 15: Conclusions
**Say:** In conclusion, sexism detection is hard because the text is often ambiguous. Standard models struggle with this. By introducing N1 to focus on hard cases, N2 to enforce logic, and N3 to soften strict labels on gray areas, we achieved higher accuracy, better logical consistency, and more reliable confidence scores.
**Emphasize:** Small, smart changes to the loss function can drastically improve how a model handles real-world nuance.
**If asked:** The biggest takeaway is that handling ambiguity explicitly is better than ignoring it.

# Slide 16: Methodology Note
**Say:** As a brief note on methodology, we originally planned to fine-tune a massive Gemini model for this task. However, to stay within free-tier limits, we pivoted to using pre-extracted MPNet embeddings and training a smaller, highly efficient neural network on top of them. This proved that you don't need massive compute to do innovative research!
**Emphasize:** We achieved these results using efficient, accessible methods.
**If asked:** MPNet is a highly regarded sentence embedding model that is much cheaper to run than a full LLM.

# Slide 17: Q&A
**Say:** Thank you for listening! I'd now be happy to take any questions you have about the dataset, our novel loss functions, or our experimental results.
**Emphasize:** (Smile and be ready to reference previous slides).
**If asked:** N/A

## Likely Q&A

**Q1: What exactly is an embedding?**
A: Think of it like a barcode for meaning. It turns a sentence into a list of numbers so that sentences with similar meanings have similar numbers. We used a pre-trained model to generate these.

**Q2: Why didn't you just fine-tune a large model like BERT or Gemini?**
A: We hit free-tier rate limits with Gemini. Using pre-computed embeddings and a smaller neural network allowed us to iterate quickly and focus on our novel loss functions without needing massive computing power.

**Q3: What does F1 score mean and why use it?**
A: F1 is a score from 0 to 1 that balances precision (not crying wolf) and recall (catching everything). It's better than plain accuracy when you have uneven data (e.g., if only 20% of tweets are sexist).

**Q4: Can you explain again how N1 focuses on hard examples?**
A: If the model predicts 0.5 (a total guess), N1 mathematically multiplies the penalty for that tweet, forcing the model to update its weights more. If it predicts 0.99 and is right, the penalty is tiny, so it ignores it.

**Q5: Why is it bad if a model is "overconfident"?**
A: If a system flags a borderline tweet as 99% definitely sexist, a human moderator might blindly trust it. We want the model to say "I'm only 60% sure," so humans know to review it carefully. This is called calibration.

**Q6: What is a logical violation in N2?**
A: It's when the model says "There is only a 10% chance this tweet is sexist at all," but simultaneously says "There is a 80% chance this tweet contains stereotyping." That makes no sense. N2 penalizes this exact scenario.

**Q7: Why did N1+N3 work best for Task 1, but N1 alone for Task 2?**
A: Task 1 is a broader, fuzzier concept (sexist vs not), so N3's label smoothing helped with the ambiguity. Task 2 has 5 specific categories, which might require stricter boundaries where smoothing isn't as helpful.

**Q8: How did you pick the numbers like alpha=0.5 or gamma=2?**
A: These are called hyperparameters. We picked standard starting values based on previous research (like the original Focal Loss paper) and tuned them slightly to see what worked best for our data.

**Q9: Does N3 change the actual dataset labels?**
A: No, it only temporarily softens the target *during training*, and only for tweets the model is currently struggling with. The actual ground-truth dataset stays the same.

**Q10: What is Expected Calibration Error (ECE)?**
A: It measures if the model's confidence matches reality. If a model says it's 80% confident on 100 tweets, exactly 80 of them should be right. ECE measures how far off it is from that ideal.

**Q11: What is an ablation study?**
A: It's like taking parts out of a car engine one by one to see what each part actually does. We tested every combination of N1, N2, and N3 to prove they each contribute something unique.

**Q12: What's next for this project?**
A: We'd love to test these loss functions on other subjective tasks, like hate speech or misinformation detection, where ambiguity and disagreement are also huge problems.
