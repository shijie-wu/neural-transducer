#!/bin/bash
for lng in aka ang ast aze azg bak ben bod cat ceb cly cpa cre crh ctp czn dak dan deu dje eng est evn fas fin frm frr fur gaa glg gmh gml gsw hil hin isl izh kan kaz kir kjh kon kpv krl lin liv lld lud lug mao mdf mhr mlg mlt mwf myv nld nno nob nya olo ood orm ote otm pei pus san sme sna sot swa swe syc tel tgk tgl tuk udm uig urd uzb vec vep vot vro xno xty zpv zul; do
    python example/sigmorphon2020-shared-tasks/augment.py task0-data/original $lng --examples 10000
done
