# Proxy "new photo" set review notes (12-class garbage dataset, web photos)

Default: keep candidate with proposed bin. Exclude photos that are not a disposal scenario (produce growing on a
plant, market piles, product catalog shots of many items), photos with no single main item, and items whose bin
needs information not visible in the photo.

## Sheet 00 (O000-O019, biological)
- exclude: O004 (fruit growing on tree), O005 (market pile of fruit), O013 (pile of zucchini, market), O018 (berries on plant).
- keep: all others (eaten corn cob, rotten fruit, bread, pasta, pizza slices O003, plate of fried rice O019).

## Sheet 01 (O020-O039, biological)
- exclude: O022, O024, O030, O031, O033, O034, O036, O037, O038 (growing produce, market piles, still lifes, screenshot).
- keep: O020 (KFC bones), O021, O023, O025-O029, O032, O035, O039 (food, leftovers, scrap piles).

## Sheet 02 (O040-O047 biological; O048-O061 cardboard)
- exclude: O040, O043, O044, O045, O046 (art/market/growing); O050 (dark sheet unclear); O054 (shelf of boxes).
- keep: O041, O042, O047; cardboard O048, O049, O051-O053, O055-O059.

## Sheet 03 (O060-O061 cardboard; O062-O075 paper; O076-O079 plastic)
- exclude: O061 (render), O068 (metallic-coated paper).
- keep: all others.

## Sheet 04 (O080-O091 plastic; O092-O099 metal)
- relabel -> landfill: O080 (film wrapper), O083, O084 (plastic bags).
- exclude: O081 (6 bags), O085, O086 (multiple/reusable bottles), O087 (unclear), O096, O097 (used foil), O099 (render).
- keep: O076-O079, O082, O088-O091 (rigid plastic); O092-O095, O098 (cans/lids).

## Sheet 05 (O100-O105 metal; O106-O119 glass)
- exclude: O101 (decorative letter), O102, O104 (product sets), O109, O111, O114 (multiple bottles).
- keep: O100, O105 (clean foil trays), O103 (can); all other glass -> landfill.

## Sheet 06 (O120-O129 glass; O130-O139 trash)
- exclude: O127 (frosted bottle, material unclear), O130 (brown paper, towel vs bag), O135, O138, O139 (mask products).
- relabel -> compost: O137 (apple core).
- keep: glass O120-O126, O128, O129; trash O131 (chip bag), O132, O133, O134 (ziplock), O136.

## Sheet 07 (O140-O153 trash; O154-O159 clothes)
- exclude: O148 (many brush heads), O149 (unclear tile), O150 (electric toothbrush, e-waste), O151 (stack of diapers).
- keep: masks, toothbrushes, wrappers, diaper O146/O152, clothes.

## Sheet 08 (O160-O173 clothes/shoes)
- all keep (landfill).

## Extra biological sheet (O174-O193)
- see below.
- exclude: O174, O175, O179, O181, O182, O188, O192, O193 (packaged/hand-held/market/still-life shots).
- keep: O176, O177, O178, O180, O183-O187, O189, O190, O191 (rotten fruit, scrap piles, bread, cut produce).
