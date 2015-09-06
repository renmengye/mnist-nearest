local os = require('os')
local torch = require('torch')
local LoggerClass = torch.class('Logger')

----------------------------------------------------------------------
function Logger:__init(owner, filename)
    self.owner = owner
    self.filename = filename
end

----------------------------------------------------------------------
function Logger.typeString(typ)
    if typ == 0 then
        return 'INFO'
    elseif typ == 1 then
        return 'WARNING'
    elseif typ == 2 then
        return 'ERROR'
    else
        return 'UNKNOWN'
    end
end

----------------------------------------------------------------------
function Logger.timeString(time)
    local date = os.date('*t', time)
    return string.format('%04d-%02d-%02d %02d:%02d:%02d',
        date.year, date.month, date.day, 
        date.hour, date.min, date.sec)
end

----------------------------------------------------------------------
function Logger:log(typ, text, verboseLevel)
    print(string.format(
        '%s: %s %s: %s',
        self.typeString(typ),
        self.timeString(os.time()),
        self.owner, text))
end

----------------------------------------------------------------------
function Logger:logInfo(text)
    self:log(0, text, 0)
end

----------------------------------------------------------------------
return Logger
