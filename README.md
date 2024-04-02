# cs107-FinalProject

<!--[![Build Status](https://travis-ci.com/CS107-gharvar4d/cs107-FinalProject.svg?token=Yaj31swkEpNhrSRpktqZ&branch=master)](https://travis-ci.com/CS107-gharvar4d/cs107-FinalProject) -->

[![codecov](https://codecov.io/gh/CS107-gharvar4d/cs107-FinalProject/branch/master/graph/badge.svg?token=8GK9WUDOZP)](undefined)

### Introduction
ADG4 is a package which computes a value while using automatic differentiation to compute derivatives of that value. It integrates a set of techniques to evaluate the derivative of various mathematical expressions.

Being able to take derivatives has many important practical applications across a variety of domains. In physics, the first order derivative tells us the rate of change, while further orders can tell us more pieces of information like acceleration and actergy. In statistics, applications include Bayesian inference and the training of neural networks. In economics, taking the derivative of profit and utility functions allows agents to maximize their expected outcomes. These are only a few instances wherein differentiating complex functions is useful and necessary.

Given the broad set of applications across domains, we hope ADG4 can be a useful tool that can facilitate quick and easy numerical evaluation of derivatives through computing. In our toolbox, vector calculation is supported. The tool box is capable of returning the Jacobian of a function with arbitrary dimension, and is also capable of returning the partial derivative for certain variables. We integrate both forward mode and reverse mode, and the users are welcomed to choose the mode best suited to their scientific problem. Generally speaking, we expect that forward mode is more suitable with lower dimensional input with higher dimensional output, and the contrary is true for reverse mode.

### Documentation

**[English](docs/documentation.md)** | [Español](docs/documentation-es.md) <!-- l10n:select -->


Group number: 28

Group mumbers:  

Rodrigo	Vargas	rov406@g.harvard.edu

Xin	Wei	davidwei@g.harvard.edu

Kristen	Grabarz	kgrabarz@g.harvard.edu

Boer	Zhang	boerzhang@g.harvard.edu

# ADG4 Impact and Inclusivity. 

It is important to consider the potential effects on building software and make it available to all without distinction. Our philosophy is to distribute software to anyone to any purpose and make efforts to develop it in a collaborative public matter. A good start for this is the fact that all of our source code resides in Github, an open platform where everyone can collaborate. Further, we believe that individuals across borders and languages should be able to collaborate on and use our project with equal access. We thus include documentation for non-English speakers such as Spanish, with more to come in the future.

Our group accepts two common workflows for collaboration:

1. Basic Shared Repository

Clone our repo and update with `git pull origin master`, then create a working branch with `git checkout -b MyNewBranch` and make any changes to it before staging. 
Commit locally and upload the changes (including your new branch) to GitHub with `git push origin MyNewBranch`

Then, navigate to main on GitHub where you should now see your new branch. Click on “Pull Request” button and “Send Pull Request”

2. Fork repo and pull

We can assign rights to “Collaborators”. Even though collaborators do not have push access to upstream, we accept Pull Requests (PRs) from them, reviews and then merge changes into main repo if approved.

All ADG4 collaborators should adhere to NeurIPS standards, remain vigilant and assest the impact of their code for unethical behaviour or illegal use. Some software applications where this might happen are security or privacy. If any misuse is known the project leader should be contacted immediately. We also want to ensure that our software is welcoming to individuals across areas of applied computing. With this in mind, we encourage collaborators to avoid using certain computing terms that can be considered offensive or exclusionary by some groups. A list of such terms and suggested alternatives can be found on the [Association for Computing Machinery](https://www.acm.org/diversity-inclusion/words-matter) website. Some examples include avoiding gendered pronouns and racially charged words such as blacklist/whitelist.

Finally, we are aware of [the gender gap in open source contributions](https://medium.com/intuit-engineering/open-source-where-are-the-women-ae20623529ca) and want to personally encourage female developers to collaborate on our software if they are interested in doing so. Since [data](https://www.techrepublic.com/article/diversity-why-open-source-needs-to-work-on-it-in-2020/) has indicated that access to resources would make women more likely to contribute to open-source projects, we would be happy to provide any additional materials or instructions personally if the materials contained in this repository are insufficient.

