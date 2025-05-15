# cloudytile

NOTE: this repo is under active development...

`cloudytile` is a framework (with deep learning and classical cv options) for determining if a tile from a satellite imagery stack is too cloudy to be useful.  Because 'useful' is dependent on the specific use case, this framework provides flexibility for the user to determine the best algorithm for the use case at hand.

The end use of these classifications is as a soft input to a more complicated classification problem (e.g., see [YaoGroup/lake-vision](https://github.com/YaoGroup/lake-vision)), which can guide that network to pay more attention to the image tiles that are 'useful'.

## Tile Examples
<table>
  <tr>
    <th></th>
    <th> </th>
    <th> </th>
  </tr>
  <tr>
    <th align="right">useful</th>
    <td><img src="assets/eg_useful1.png" alt="useful 1" width="240"/></td>
    <td><img src="assets/eg_useful3.png" alt="useful 2" width="240"/></td>
  </tr>
  <tr>
    <th align="right">not useful</th>
    <td><img src="assets/eg_useless1.png" alt="not useful 1" width="240"/></td>
    <td><img src="assets/eg_useless2.png" alt="not useful 2" width="240"/></td>
  </tr>
</table>

Note that even though the second example of a 'useful' tile is notably cloudy, there is still useful information about the presence of a lake (e.g., we can see through the thin cloud layer and determine that the tile has a lake, and its rough extent).  This information is useful in downstream applications, such as lake detection and drainage classification.  If we were to use a strict cloud cut or other methods, we may erroneously ignore the utility of this tile.


## Model Results
<img src="assets/training.png" alt="Training Metrics" width="480px" />
<img src="assets/confmats.png" alt="Confusion Matrices" width="480px" />

