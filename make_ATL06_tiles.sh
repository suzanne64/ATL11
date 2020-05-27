#! /usr/bin/env bash

hemisphere=$1

cycle_dir=$2

[ -d tiles ] || mkdir tiles

[ -d $cycle_dir/index ] || mkdir $cycle_dir/index
workers_started=0
if [ -f $cycle_dir/index/GeoIndex.h5 ] ; then
    echo "$cycle_dir/index exists, skipping"
else
    echo "Indexing individual ATL06es for $cycle_dir"

    > file_queue.txt
    for file in $cycle_dir/*ATL06*.h5; do
	this_file_index=$cycle_dir/index/`basename $file`
	[ -f $this_file_index ] && continue
	echo "index_glob.py -t ATL06 -H $hemisphere --index_file $this_file_index -g $file --dir_root `pwd`/$dir/" >> file_queue.txt
    done

    pboss.py -s file_queue.txt -j 8 -w -p
    workers_started=8
    echo "Making a collective ATL06 index for $cycle_dir"
    index_glob.py --dir_root=`pwd`/$cycle_dir/index/ -t h5_geoindex -H $hemisphere --index_file $cycle_dir/index/GeoIndex.h5 -g "`pwd`/$cycle_dir/index/*ATL06*.h5" --Relative

fi
    
cycle_tile_dir=tiles/$cycle_dir
[ -d $cycle_tile_dir ] || mkdir $cycle_tile_dir
echo "making a queue of indexing commands for $cycle_dir"
# make a queue of tiles
make_tiles.py -H $hemisphere -i $cycle_dir/index/GeoIndex.h5 -W 100000 -t ATL06 -o $cycle_tile_dir -q tile_queue.txt -j ATL06_field_dict.json
# run the queue
echo "running the queue for $cycle_dir"
if $workers_started == 0; then
    pboss.py -s tile_queue.txt -r -w -j 8
else
    pboss.py -s tile_queue.txt -r -w 
fi

echo "indexing tiles for $cycle_dir"
pushd $cycle_tile_dir
index_glob.py -H $hemisphere -t indexed_h5 --index_file GeoIndex.h5 -g "E*.h5" --dir_root `pwd` 
popd


#python3 geoindex_test_plot.py $cycle_tile_dir/GeoIndex.h5
