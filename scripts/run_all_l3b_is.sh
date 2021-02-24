#!/bin/bash
exec 2>&1
set -eu
# Executes l3b_is algorithm, atlas_meta, atl11_qa_util, ATL11 browse programming
# With set -eu, any fail or missing variable will exit with error, including missing (required) arguments.

THIS_SCRIPT=`basename $0`
ASAS_BIN=/att/nobackup/project/icesat-2/bin

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--rgt) rgt="$2"; shift ;;
        -n|--region) region="$2"; shift ;;
        -a|--atl06_datapath) atl06_datapath="$2"; shift ;;
        -G|--geoindex_path) geoindex_path="$2"; shift ;;
        -R|--release) release="$2"; shift ;;
        -V|--version) version="$2"; shift ;;
        -o|--output_path) output_path="$2"; shift ;;
        -s|--start_cycle) start_cycle="$2"; shift ;;
        -e|--end_cycle) end_cycle="$2"; shift ;;
        -H|--hemisphere) hemisphere="$2"; shift ;;
        -m|--dem_mosaic) dem_mosaic="$2"; shift ;;
        -c|--ctl_file) ctl_file="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Clean up any lead 0s from variables, for correct printf usage
let rgt="10#$rgt"
let region="10#$region"
let release="10#$release"
let version="10#$version"
let start_cycle="10#$start_cycle"
let end_cycle="10#$end_cycle"

atl11_outfile=`printf "%s/ATL11_%04d%02d_%02d%02d_%03d_%02d.h5" $output_path $rgt $region $start_cycle $end_cycle $release $version`
logfile=`printf "logs/ATL11_%04d%02d_%02d%02d_%03d_%02d.log" $rgt $region $start_cycle $end_cycle $release $version`
if [ ! -e logs ]; then
  mkdir logs
fi
if [ -e $logfile ]; then
  rm -f $logfile
fi

echo "Start processing: `date`" | tee -a $logfile
ALL_RES=0
if [ ! -e $atl11_outfile ]; then
  $PYTHONPATH/ATL11/ATL06_to_ATL11.py $rgt $region --cycles $start_cycle $end_cycle -d "$atl06_datapath" -R $release -V $version -o $output_path -H $hemisphere -G "$geoindex_path" --verbose  | tee -a $logfile
  RES=${PIPESTATUS[0]}
  if [ ${RES} -ne 0 ] ; then
    echo "${THIS_SCRIPT} Warning: ATL06_to_ATL11.py did not complete successfully"
    echo "${THIS_SCRIPT} Warning: ATL06_to_ATL11.py did not complete successfully" | tee -a $logfile
    ALL_RES=3
    echo "Exit code: "$ALL_RES | tee -a $logfile
    exit $ALL_RES
  else
    echo "${THIS_SCRIPT} - ATL06_to_ATL11.py completed successfully."
  fi
  echo " "

  if [ ! -e $atl11_outfile ]; then
    echo "ATL11 output not created"
    exit 3
  fi
fi

$ASAS_BIN/atlas_meta $ctl_file | tee -a $logfile
RES=${PIPESTATUS[0]}
if [ ${RES} -ne 0 ] ; then
  echo "${THIS_SCRIPT} Warning: atlas_meta did not complete successfully"
  echo "${THIS_SCRIPT} Warning: atlas_meta did not complete successfully" | tee -a $logfile
  ALL_RES=3
else
  echo "${THIS_SCRIPT} - atlas_meta completed successfully."
fi
echo " "

if [ ! -e BRW_template.h5 ]; then
  ln -s $ASAS_BIN/BRW_template.h5 .
fi
python3 $PYTHONPATH/ATL11/ATL11_browse_plots.py $atl11_outfile -H $hemisphere -m $dem_mosaic | tee -a $logfile
RES=${PIPESTATUS[0]}
if [ ${RES} -ne 0 ] ; then
  echo "${THIS_SCRIPT} Warning: ATL11_browse_plots.py did not complete successfully"
  echo "${THIS_SCRIPT} Warning: ATL11_browse_plots.py did not complete successfully" | tee -a $logfile
  ALL_RES=3
else
  echo "${THIS_SCRIPT} - ATL11_browse_plots.py completed successfully."
fi
echo " "

$ASAS_BIN/atl11_qa_util $ctl_file | tee -a $logfile
RES=${PIPESTATUS[0]}
if [ ${RES} -ne 0 ] ; then
  echo "${THIS_SCRIPT} Warning: atl11_qa_util did not complete successfully"
  echo "${THIS_SCRIPT} Warning: atl11_qa_util did not complete successfully" | tee -a $logfile
  ALL_RES=3
else
  echo "${THIS_SCRIPT} - atl11_qa_util completed successfully."
fi
echo " "
echo "End processing: `date`" | tee -a $logfile
echo "Exit code: "$ALL_RES | tee -a $logfile
exit $ALL_RES

