# Get SPD REAL275 network
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1p72NdY4Bie_sra9U8zoUNI4fTrQZdbnc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1p72NdY4Bie_sra9U8zoUNI4fTrQZdbnc" -O deformnet_eval.zip && rm -rf /tmp/cookies.txt
unzip deformnet_eval.zip
mkdir results
mv deformnet_eval/* ./results
rmdir deformnet_eval
rm deformnet_eval.zip
