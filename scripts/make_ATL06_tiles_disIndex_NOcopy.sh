#! /usr/local/bin/bash -vx
# #! /usr/bin/env bash

hemisphere=$1

cycle_dir=$2
release=$3
script_path=$PYTHONPATH

echo "Start time `date`"

cycle=`echo $cycle_dir | awk -F '/' '{print $(NF)}' | awk -F '_' '{print $2}'`
if [ ! -d $cycle_dir ]; then
    project_base=$4
    cycle=`echo $cycle_dir | awk -F '/' '{print $(NF)}' | awk -F '_' '{print $2}'`
#    [ -d $project_base/ATL06_copy ] || mkdir $project_base/ATL06_copy
    if [ $hemisphere == 1 ]; then
	hemistring='Arctic'
	regions=( 03 04 05 )
    else
	hemistring='Antarctic'
	regions=( 10 11 12 )
    fi
#    [ -d $project_base/ATL06_copy/$hemistring ] || mkdir $project_base/ATL06_copy/$hemistring
#    [ -d $project_base/ATL06_copy/$hemistring/$release ] || mkdir $project_base/ATL06_copy/$hemistring/$release
    [ -d $cycle_dir ] || mkdir $cycle_dir
    for reg in "${regions[@]}"
    do
#         cp -a /css/icesat-2/ATLAS/ATL06.${release}/*/ATL06_*_????${cycle}${reg}_${release}_*.h5 $cycle_dir/.
         ln -s /css/icesat-2/ATLAS/ATL06.${release}/*/ATL06_*_????${cycle}${reg}_${release}_*.h5 $cycle_dir/.
    done
fi

echo "Copy done time `date`"

[ -d tiles ] || mkdir tiles

[ -d $cycle_dir/index ] || mkdir $cycle_dir/index

echo "Indexing individual ATL06es for $cycle_dir"

#> file_queue.txt
> file_queue_${cycle}.txt
for file in $cycle_dir/*ATL06*.h5; do
    this_file_index=$cycle_dir/index/`basename $file`
    [ -f $this_file_index ] && continue
    echo $file
#    echo "${script_path}/pointCollection/scripts/index_glob.py -t ATL06 -H $hemisphere --index_file $this_file_index -g $file --dir_root `pwd`/$dir/" >> file_queue_${cycle}.txt
    echo "${script_path}/pointCollection/scripts/index_glob.py -t ATL06 -H $hemisphere --index_file $this_file_index -g `readlink $file` --dir_root `pwd`/$dir/" >> file_queue_${cycle}.txt
done

exit

#pboss.py -s file_queue.txt -j 8 -w -p
#parallel < file_queue.txt
#cat file_queue.txt | parallel --ssh "ssh -q" --workdir . --env PYTHONPATH -S icesat102,icesat103,icesat104,icesat105,icesat106,icesat107,icesat108,icesat109,icesat110,icesat111
#cat file_queue.txt | parallel --ssh "ssh -q" --workdir . --env PYTHONPATH -S icesat102,icesat103,icesat104,icesat106,icesat107,icesat109,icesat110,icesat111
#tmux new-session -d -s cycle "pboss.py -s file_queue_${cycle}.txt -r"
#declare -a squeue_ids=()
#for i in `seq 1 $nnodes`
#do
#  squeue_ids+=(`sbatch sc_14_workers | awk '{print $NF}'`)
#done

echo "File indexing done time `date`"

echo "Making a collective ATL06 index for $cycle_dir"
#index_glob.py --dir_root=`pwd`/$cycle_dir/index/ -t h5_geoindex -H $hemisphere --index_file $cycle_dir/index/GeoIndex.h5 -g "`pwd`/$cycle_dir/index/*ATL06*.h5" -v --Relative
#${script_path}/pointCollection/scripts/index_glob.py --dir_root=`pwd`/$cycle_dir/index/ -t h5_geoindex -H $hemisphere --index_file $cycle_dir/index/GeoIndex.h5 -g "$cycle_dir/index/*ATL06*.h5" -v --Relative

trimcycle_dir=`echo $cycle_dir | awk -F'/' '{print $(NF-2)"/"$(NF-1)"/"$(NF)}'`
cycle_tile_dir=tiles/$trimcycle_dir
[ -d $cycle_tile_dir ] || mkdir -p $cycle_tile_dir
echo "making a queue of indexing commands for $cycle_dir"
# make a queue of tiles
${script_path}/pointCollection/scripts/make_tiles.py -H $hemisphere -i $cycle_dir/index/GeoIndex.h5 -W 100000 -t ATL06 -o $cycle_tile_dir -q tile_queue_${cycle}.txt -j ${script_path}/ATL11/ATL06_field_dict.json
# run the queue

exit

echo "running the queue for $cycle_dir"
#pboss.py -s tile_queue.txt -r -w
#parallel < tile_queue.txt
#cat tile_queue.txt | parallel --ssh "ssh -q" --workdir . --env PYTHONPATH -S icesat102,icesat103,icesat104,icesat105,icesat106,icesat107,icesat108,icesat109,icesat110,icesat111
#cat tile_queue.txt | parallel --ssh "ssh -q" --workdir . --env PYTHONPATH -S icesat102,icesat103,icesat104,icesat106,icesat107,icesat109,icesat110,icesat111
tmux new-session -d -s tiles_${cycle} "pboss.py -s tile_queue_${cycle}.txt -r"
declare -a squeue_ids=()
nnodes=5
for i in `seq 1 $nnodes`
do
  squeue_ids+=(`sbatch sc_14_workers | awk '{print $NF}'`)
done

echo "tile generation done time `date`"
echo "indexing tiles for $cycle_dir"
pushd $cycle_tile_dir
${script_path}/pointCollection/scripts/index_glob.py -H $hemisphere -t indexed_h5 --index_file GeoIndex.h5 -g "E*.h5" --dir_root `pwd` -v 
popd


#python3 geoindex_test_plot.py $cycle_tile_dir/GeoIndex.h5

echo "Finished processing for hemisphere $hemisphere and cycle_dir $cycle_dir"
echo "Stop time `date`"
