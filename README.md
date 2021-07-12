Code and self-explanations as I work through [Deep Learning with Pytorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf#G1.1016757).
The data was obtained from [this repository](https://github.com/deep-learning-with-pytorch/dlwpt-code), which accompanies the book.

The code implementations are only upto chapter 12, after which it got a little dry, so I just read about what they were trying to do, and atempted to understand it.

In all, I really enjoyed this book! It has lot of valuable info and advice, beyond just teaching you about pytorch. Definitely see myself coming back to it and referring to certain sections if I do some projects in the future. The end of chapter 14 on improvements and a realistic view into the field was a goldmine.

### cool little takeaways

- Too little structure, and it will become difficult to perform experiments cleanly, troubleshoot problems, or even describe what you’re doing! Conversely, too much structure means you’re wasting time writing infrastructure that you don’t need and most likely slowing yourself down by having to conform to it after all that plumbing is in place.
- Plus it can be tempting to spend time on infrastructure as a procrastination tactic, rather than digging into the hard work of making actual progress on your project. Don’t fall into that trap!
- A good way to structure the main model is to have it work with a shell (and accept command line arguments), as well as allow it to be imported into a jupyter notebook.
- We’ve known that our model’s performance was garbage since chapter 11. If our metrics told us anything but that, itwould point to a fundamental flaw in the metrics!
- When done properly, augmentation can increase training set size beyond what the model is capable of memorising, resulting in the model being forced to increasinglt rely on generalization, which is exactly what we want.
- flask is not too good at serving torch models. check out [sanic](https://sanicframework.org/en/)

### setup in new env (at time of writing)

```bash
pip install torch torchvision matplotlib diskcache tensorflow jupyter nbconvert seaborn SimpleITK
```

- tensorboard didnt work, so i reinstalled everything in a new environment and its coolll
