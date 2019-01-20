---
layout: post
title: "Getting started with Jekyll on Windows platform"
description: "Setting-up Jekyll while using windows"
img: jekyll-poster.png
date: 2017-09-07 18:20 +0200
---

There are many blogs and videos on how to setup Jekyll on windows. But it's quite confusing and complex to set it up on windows especially if someone doesn't have a good knowledge of Linux and Jekyll. The problem arises because of version incompatibility and windows platform where Jekyll is officially not supported. A lot of relevant material is older than an year. While using templates and themes, often it becomes hard to deal with the gems and bundle (if someone is unfamiliar with ruby and rails). This post will help you use this powerful static website development tool and set it up on windows without hassle. 

First, I would like to point out a specific video which is quite simple to follow and will surely make your life easier: https://www.youtube.com/watch?v=BTX_uh_v99I

Secondly, there are a lot of templates which can be used to develop a beautiful website with powerful functionalities. Check them out here: http://themes.jekyllrc.org/

Some points for saving time during setup:
1. Follow all the steps precisely and remember to restart the PowerShell as admin every time it is mentioned.
2. In case the bundle is not latest and when you try to use already existing templates (may be from http://themes.jekyllrc.org/ ) it may throw gem not found error. In such cases force the bundle to switch to the latest version.

3. Here's another useful video in case you are interested in using templates and facing problems in setting them up. The most common problems that I have noticed are related to relative path and slash '/'.
https://www.youtube.com/watch?v=bty7LHm14CA

4. It's worth understanding the basic functionality of Jekyll and how it is used with GitHub, especially if you are a developer or even if you are trying to create a basic portfolio.
https://www.youtube.com/watch?v=SWVjQsvQocA

5. If your localhost (with Jekyll it's 4000 by default) turns out to be a blank page then clear cache and restart computer before trying anything funny like reinstalling rube or gems.

6. In case you try a template and 'jekyll serve' command is giving an error due to mismatch in versions, the best option is to try 'bundle exec Jekyll serve'.

Hope this helps!
