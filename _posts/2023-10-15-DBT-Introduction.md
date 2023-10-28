---
layout: post
title: DBT Introduction
date: 2023-10-14 21:39:00-0400
description: What is DBT
tags: formatting data_engineer
categories: sample-posts
giscus_comments: true
related_posts: false
---

## Introduction

As a data team leader, I encounter frequently problems related to data management. One of the challenging problems is how to manage data aggregation flows in a effective way. We have difficulty when the team size become bigger, and many projects need to build so we need a tool that could help us control effectively in building data processing flows. We often encounter bellow things:
- Don't know what data source are used to build or aggregate a table
- Our data is easily broken down when another data source has some issues
- We are not able to reuse computations on aggregating analytic data

Finally, we found a fantastic tool that is Data Build Tool (DBT).

## What's Data Build Tool

Data Build Tool (DBT) is an open-source tool that allows data engineers (DE),
data analysts (DA), or event data analytics engineer (DAE) to transform data
in their data warehouses. It simplifies the process of transforming raw data into usable tables and views.
