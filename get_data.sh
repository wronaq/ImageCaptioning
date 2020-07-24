#! /usr/bin/bash

echo '---- Starting. It will take some time. ----'

# make structure
mkdir /opt/cocoapi
mkdir /opt/cocoapi/images
mkdir /opt/cocoapi/annotations
echo '1. Dir structure made in /opt. Starting download train data.'

# download files
wget 'http://images.cocodataset.org/zips/train2014.zip'
echo '2. Train dataset downloaded. Starting download validation data.'
wget 'http://images.cocodataset.org/zips/val2014.zip'
echo '3. Validation dataset downloaded. Starting download annotations.'
wget 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
echo '4. Annotations downloaded. Unziping.'

# unzip
unzip train2014.zip -d /opt/cocoapi/images
echo '5. Train dataset unziped.'
unzip val2014.zip -d /opt/cocoapi/images
echo '6. Validation dataset unziped.'
unzip annotations_trainval2014.zip -d /opt/cocoapi
echo '7. Annotations unziped. Deleting redundant files.'

# rename
mv /opt/cocoapi/images/val2014 /opt/cocoapi/images/val2014.old
mv /opt/cocoapi/annotations/captions_val2014.json /opt/cocoapi/annotations/captions_val2014.json.old

# delete
rm -f train2014.zip val2014.zip annotations_trainval2014.zip /opt/cocoapi/annotations/person*
echo '8. Redundant files removed.'

echo '---- All done. ----'
