local Logger = require('logger')
local logger = Logger()

local imageqa = {}
function imageqa.readDict(filename)
    fh, err = io.open(filename)
    if err then
        logger:logError(string.format('Error reading %s', filename))
        return
    end
    while true do
        line = fh:read()
        if line == nil then
            break
        end
        print(line)
    end
    fh:close()
end

-- function imageqa.decodeSentence(ids, dict)
-- end

imageqa.readDict('../data/cocoqa/answer_vocabs.txt')

return imageqa
