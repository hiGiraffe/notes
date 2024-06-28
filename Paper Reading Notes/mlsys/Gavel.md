# Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads

[Deepak Narayanan](https://deepakn94.github.io/#about)在OSDI 2020发表的一篇工作，先前其在Stanford，现在在Nvidia

## 总结概述

在调度DL training任务时考虑不同加速器的异构性，并且将传统的调度策略建模成一个优化问题，通过求解优化问题得到最优的资源分配方式。

## 先前工作存在问题

随着时间发展，一个系统会积累很多的加速器类型，如何在考虑公平性或makespan的情况下给多用户分配资源存在困难

## 难点

1. Performance Heterogeneity

​		不同的任务适合的加速器可能是不一样的。为了满足SLO，可能调度到资源并不适合任务。

2. Generality across Policies

​		目前的调度可以支持分层调度，在不同层级进行调度。但一些新的工作，假如关注公平性后，可能不能轻易地使用以上的调度策略

3. Colocation and Placement Optimizations.

​		目前相关工作有空间共享和位置敏感性，考虑了性能感知后，这些优化可以获得更好的性能。

## 解决方案

在pdf中，To be continue