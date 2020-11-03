run=bash

for lang in aka ang ast aze azg bak ben bod cat ceb cly cpa cre crh ctp czn dak dan deu dje eng est evn fas fin frm frr fur gaa glg gmh gml gsw hil hin isl izh kan kaz kir kjh kon kpv krl lin liv lld lud lug mao mdf mhr mlg mlt mwf myv nld nno nob nya olo ood orm ote otm pei pus san sme sna sot swa swe syc tel tgk tgl tuk udm uig urd uzb vec vep vot vro xno xty zpv zul; do
    # regular data
    echo trm $lang
    $run example/sigmorphon2020-shared-tasks/task0-trm.sh $lang
    echo mono $lang
    $run example/sigmorphon2020-shared-tasks/task0-mono.sh $lang

    # augmented data
    echo trm hall $lang
    $run example/sigmorphon2020-shared-tasks/task0-hall-trm.sh $lang
    echo mono hall $lang
    $run example/sigmorphon2020-shared-tasks/task0-hall-mono.sh $lang
done

for lang in afro-asiatic austronesian dravidian germanic indo-aryan iranian niger-congo oto-manguean romance turkic uralic; do
    # regular data
    echo trm $lang
    $run example/sigmorphon2020-shared-tasks/task0-trm.sh $lang
    echo mono $lang
    $run example/sigmorphon2020-shared-tasks/task0-mono.sh $lang

    # augmented data
    echo trm hall $lang
    $run example/sigmorphon2020-shared-tasks/task0-hall-trm.sh $lang
    echo mono hall $lang
    $run example/sigmorphon2020-shared-tasks/task0-hall-mono.sh $lang
done
