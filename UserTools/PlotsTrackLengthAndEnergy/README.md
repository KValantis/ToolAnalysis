# PlotsTrackLengthAndEnergy

The `PlotsTrackLengthAndEnergy` tool is used to plot the reconstructed track length and energy from the DNN and the BDT. Therefore this tool should be run only after running the `EnergyRecoPredict` toolchain. This tool loads the `EnergyReco` boost store that is being saved locally by the `BDTMuonEnergyPredict` tool in order to make the plots we want.

## Data

Describe any data formats PlotsTrackLengthAndEnergy creates, destroys, changes, or analyzes. E.G.

**RawLAPPDData** `map<Geometry, vector<Waveform<double>>>`
* Takes this data from the `ANNIEEvent` store and finds the number of peaks


## Configuration

Describe any configuration variables for PlotsTrackLengthAndEnergy.

```
param1 value1
param2 value2
```
