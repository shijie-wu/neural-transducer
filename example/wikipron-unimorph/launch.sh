run=qsub
# run the monotonic hard attention model
for lang in bul ell eng_uk eng_us fre hun lit pol rus; do
echo notags-mono $lang
$run example/wikipron-unimorph/notags-mono.sh $lang
echo tags-mono $lang
$run example/wikipron-unimorph/tags-mono.sh $lang
done
# run the transformer models
for lang in bul ell eng_uk eng_us fre hun lit pol rus; do
echo notags-trm $lang
$run example/wikipron-unimorph/notags-trm.sh $lang
echo tags-trm $lang
$run example/wikipron-unimorph/tags-trm.sh $lang
done