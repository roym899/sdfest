wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fxc9UoRhfTsoV3ZML3Mx_mc79904zpcx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fxc9UoRhfTsoV3ZML3Mx_mc79904zpcx" -O params.zip && rm -rf /tmp/cookies.txt
unzip params.zip
rm params.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1z6u3Oo3eza9qftoiB21YVhP3W-hovAYv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1z6u3Oo3eza9qftoiB21YVhP3W-hovAYv" -O masks_real_test.zip && rm -rf /tmp/cookies.txt
unzip masks_real_test.zip
rm masks_real_test.zip
mv masks_real_test ./dataset/
