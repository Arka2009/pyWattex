# HardPkpMin3
3rd iteration of 
Design Space Exploration and peak power minimization of malleable DAGs,
to be submitted for DATE2020/TCAD2020

1. Baseline implementation "https://github.com/PowerCapDAG/PowerCapDagScheduling.git" 
Cite the following paper.

@inproceedings{DemirciMH18,
  author    = {G{\"{o}}kalp Demirci and
               Ivana Marincic and
               Henry Hoffmann},
  title     = {A divide and conquer algorithm for {DAG} scheduling under power constraints},
  booktitle = {Proceedings of the International Conference for High Performance Computing,
               Networking, Storage, and Analysis, {SC} 2018, Dallas, TX, USA, November
               11-16, 2018},
  pages     = {36:1--36:12},
  year      = {2018},
  url       = {http://dl.acm.org/citation.cfm?id=3291704},
  timestamp = {Mon, 12 Nov 2018 09:20:44 +0100},
  biburl    = {https://dblp.org/rec/bib/conf/sc/DemirciMH18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

The algorithm have been modified re-implemented from scratch to serve as
as baseline for our DATE2020 submission.

2. Convex optimization applied to malleable DAG scheduling in-order to reduce
peak power consumption
