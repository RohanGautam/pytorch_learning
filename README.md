Code and self-explanations as I work through [Deep Learning with Pytorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf#G1.1016757).

The data was obtained from [this repository](https://github.com/deep-learning-with-pytorch/dlwpt-code), which accompanies the book.

### cool little takeaways

> Too little structure, and it will become difficult to perform experiments cleanly, troubleshoot problems, or even describe what you’re doing! Conversely, too much structure means you’re wasting time writing infrastructure that you don’t need and most likely slowing yourself down by having to conform to it after all that plumbing is in place.

> Plus it can be tempting to spend time on infrastructure as a procrastination tactic, rather than digging into the hard work of making actual progress on your project. Don’t fall into that trap!

- A good way to structure the main model is to have it work with a shell (and accept command line arguments), as well as allow it to be imported into a jupyter notebook.
