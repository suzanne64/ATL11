#! /usr/bin/env bash

[ -d tiles ] || mkdir tiles

for cycle_dir in cycle_04 cycle_03; do
    [ -d $cycle_dir/index ] || mkdir $cycle_dir/index

    echo "Indexing individual ATL06es for $cycle_dir"
    for file in $cycle_dir/*ATL06*.h5; do
	this_file_index=$cycle_dir/index/`basename $file`
    	[ -f $this_file_index ] && continue
  	  echo $file
    	  index_glob.py -t ATL06 -H 1 --index_file $this_file_index -g $file --dir_root `pwd`/$dir/
    done
    echo "Making a collective ATL06 index for $cycle_dir"
    index_glob.py --dir_root=`pwd`/$cycle_dir/index/ -t h5_geoindex -H 1 --index_file $cycle_dir/index/GeoIndex.h5 -g "`pwd`/$cycle_dir/index/*ATL06*.h5" -v --Relative

    cycle_tile_dir=tiles/$cycle_dir
    [ -d $cycle_tile_dir ] || mkdir $cycle_tile_dir
    echo "making a queue of indexing commands for $cycle_dir"
    # make a queue of tiles
    make_tiles.py -H 1 -i $cycle_dir/index/GeoIndex.h5 -W 100000 -t ATL06 -o $cycle_tile_dir -q queue.txt -j ATL06_field_dict.json
    # run the queue
    echo "running the queue for $cycle_dir"
    xargs -a queue.txt -L 1 -P 8 python3

    echo "indexing tiles for $cycle_dir"
    pushd $cycle_tile_dir
    index_glob.py -H 1 -t indexed_h5 --index_file GeoIndex.h5 -g "E*.h5" --dir_root `pwd` -v 
    popd
done

python3 geoindex_test_plot.py tiles/cycle*/GeoIndex.h5






